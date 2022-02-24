from typing import IO, List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from torch import tensor
from torch.cuda import is_available
from tokenizers.implementations import ByteLevelBPETokenizer

from language_model.domain.modeling import CUDA, CPU
from language_model.domain.modeling.data import (
    EOD_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    TOKENIZATION_MSG,
    IS_FITTED_FIELD,
    IS_FITTED_MESSAGE,
    TOKENIZER_FIELD,
    CHECK_TOKENIZER_ERROR_MESSAGE
)
from language_model.domain.modeling.data.errors import MissingTokenizerError, NotFittedDatasetError

device = CUDA if is_available() else CPU


class LanguageModelingDataset:
    def __init__(self, batch_size: int, bptt: int) -> None:
        self.batch_size = batch_size
        self.bptt = bptt

    def __len__(self) -> int:
        self._check_is_fitted()
        return self._sequence_of_ids_length

    def _check_is_fitted(self) -> None:
        if not hasattr(self, IS_FITTED_FIELD):
            raise NotFittedDatasetError(IS_FITTED_MESSAGE)

    @staticmethod
    def _add_special_tokens(text: str) -> str:
        return f"{SOS_TOKEN} {text} {EOD_TOKEN}"

    def set_tokenizer(self, tokenizer: ByteLevelBPETokenizer) -> None:
        self._tokenizer = tokenizer

    @staticmethod
    def _read_text_file(path_to_text_file: str) -> List[str]:
        with open(path_to_text_file, "r") as file:
            return file.readlines()

    def fit(self, path_to_text_file: str, vocabulary_size: int = 30000) -> None:
        self._check_tokenizer()
        self._fit_tokenizer(path_to_text_file, self._tokenizer, vocabulary_size)
        texts = self._read_text_file(path_to_text_file)
        tokenized_texts = self._tokenize_texts(texts)
        self.train_sequence_of_ids, self.test_sequence_of_ids = self._orchestrate_dataset_creation(tokenized_texts, self.batch_size, self.bptt)
        self._is_fitted = True
        self._sequence_of_ids_length = self._get_sequence_of_ids_length(self.train_sequence_of_ids)

    @staticmethod
    def _get_sequence_of_ids_length(sequence_of_ids: np.ndarray) -> int:
        return sequence_of_ids.shape[-1]

    def _check_tokenizer(self) -> None:
        if not hasattr(self, TOKENIZER_FIELD):
            raise MissingTokenizerError(CHECK_TOKENIZER_ERROR_MESSAGE)

    @staticmethod
    def _fit_tokenizer(path_to_text_file: Union[str, List[str]], tokenizer: ByteLevelBPETokenizer, vocabulary_size: int) -> None:
        tokenizer.train(path_to_text_file, vocabulary_size, special_tokens=[EOD_TOKEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN])

    def _orchestrate_dataset_creation(self, sequence_of_ids: List[int], batch_size: int, bptt: int) -> Tuple[np.ndarray, np.ndarray]:
        sequence_of_ids_length = self._get_length_of_tokenized_texts(sequence_of_ids)
        total_number_of_batches = self._get_total_number_of_batches(sequence_of_ids_length, batch_size, bptt)
        sequence_of_ids = self._truncate_sequence_of_ids_for_batch_processing(sequence_of_ids, batch_size, bptt, total_number_of_batches)
        sequence_of_ids = self._reshape_sequence_of_ids_for_batch_processing(sequence_of_ids, batch_size)
        target_sequence_of_ids = self._generate_empty_target_array(sequence_of_ids)
        target_sequence_of_ids = self._fill_target_array(sequence_of_ids, target_sequence_of_ids)
        return sequence_of_ids, target_sequence_of_ids


    @staticmethod
    def _get_length_of_tokenized_texts(sequence_of_ids: List[int]) -> int:
        return len(sequence_of_ids)


    @staticmethod
    def _get_total_number_of_batches(texts_length: int, batch_size: int, bptt: int) -> int:
        return texts_length // (batch_size * bptt)


    @staticmethod
    def _truncate_sequence_of_ids_for_batch_processing(sequence_of_ids: List[int], batch_size: int, bptt: int, total_number_of_batch: int) -> List[
        int]:
        return sequence_of_ids[0: batch_size * bptt * total_number_of_batch]


    @staticmethod
    def _reshape_sequence_of_ids_for_batch_processing(sequence_of_ids: List[int], batch_size: int) -> np.ndarray:
        return np.reshape(sequence_of_ids, (batch_size, -1))


    @staticmethod
    def _generate_empty_target_array(sequence_of_ids: np.ndarray) -> np.ndarray:
        return np.zeros_like(sequence_of_ids)


    @staticmethod
    def _fill_target_array(sequence_of_ids: np.ndarray, empty_sequence: np.ndarray) -> np.ndarray:
        empty_sequence[:, :-1] = sequence_of_ids[:, 1:]
        empty_sequence[:, -1] = PAD_TOKEN
        return empty_sequence


    def _tokenize_texts(self, texts: List[str]) -> List[int]:
        tokenized_texts = []
        for text in tqdm(texts, desc=TOKENIZATION_MSG):
            tokenized_texts.extend(self._tokenizer.encode(self._add_special_tokens(text)).ids)
        return tokenized_texts


    def get_train_batches(self, batch_index: int) -> Tuple[tensor, tensor]:
        return (
            tensor(self.train_sequence_of_ids[:, batch_index: batch_index + self.bptt], device=device, dtype=torch.long),
            tensor(self.test_sequence_of_ids[:, batch_index: batch_index + self.bptt], device=device, dtype=torch.long)
        )

    def transform_text(self, text: str) -> tensor:
        self._check_is_fitted()
        text = self._add_special_tokens(text)
        sequence_of_ids = self._tokenizer.encode(text).ids
        return tensor([sequence_of_ids], device=device, dtype=torch.long)
