import logging
from os import path
from typing import Tuple

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from torch import tensor, long

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class LanguageModelingDataset(object):
    """LanguageModelingDataset is an object that performs some data preprocessing

    This object performs some data preprocessing over a text file:
        - train tokenizer 
        - tokenize text
        - batchify text data

    Parameters
    ----------
    batch_size : int
        Size of the batch to perform gradient descent
    bptt: int
        Max sequence length to use for gradient descent
    training: bool
        Whenever to train a new tokenizer
    """
    def __init__(
        self,
        batch_size: int,
        bptt: int,
        training: bool = True,
    ) -> None:

        self.batch_size = batch_size
        self.bptt = bptt
        self.training = training

    def __len__(self) -> int:
        return self.tokenized_sequences.shape[-1]

    def _preprocess_data(self, path_to_data: str, tokenizer: Tokenizer) -> None:
        logger.info("Opening: {}".format(path_to_data))
        tokenized_sequences = []
        with open(path_to_data, "r") as text_file:
            for text in tqdm(text_file.readlines()):
                tokenized_sequences.extend(
                    tokenizer.encode(
                        "<SOS> {} <EOS>".format(text)
                    ).ids
                )
        
        logger.info("Reshaping dataset...")
        self.total_number_of_batches = len(tokenized_sequences) // (self.batch_size * self.bptt)
        self.tokenized_sequences = np.reshape(
            tokenized_sequences[0: self.total_number_of_batches * self.batch_size * self.bptt],
            (self.batch_size, -1)
        )

        self.target_tokenizer_sequences = np.zeros_like(self.tokenized_sequences)
        self.target_tokenizer_sequences[:, :-1] = self.tokenized_sequences[:, 1:]
        self.target_tokenizer_sequences[:, -1] = self.tokenized_sequences[:, 0]

    def _train_tokenizer(self, path_to_data: str, tokenizer: Tokenizer, vocabulary_size: int) -> None:
        trainer = BpeTrainer(
            vocab_size=vocabulary_size,
            continuing_subword_prefix="##",
            end_of_word_suffix="</w>"
        )

        tokenizer.train(
            trainer,
            [path_to_data]
        )

    def get_batches(self, index: int) -> Tuple[tensor, tensor]:
        return (
            tensor(self.tokenized_sequences[:, index: index + self.bptt], dtype=long),
            tensor(self.target_tokenizer_sequences[:, index: index + self.bptt], dtype=long)
        )

    def fit(self, path_to_data: str, tokenizer: Tokenizer, vocabulary_size: int) -> None:
        """Fit the object

        Parameters
        ----------
        path_to_data: str
            Path to text file
        tokenizer: Tokenizer
            Instance of Tokenizer to train on `path_to_data`
        vocabulary_size : int
            Desired size for the vocabulary
        """
        
        if self.training:
            logger.info("Training tokenizer...")
            
            self._train_tokenizer(
                path_to_data,
                tokenizer,
                vocabulary_size
            )

        self._preprocess_data(path_to_data, tokenizer)