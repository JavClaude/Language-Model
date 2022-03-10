import random
from typing import List

import torch
from torch import tensor, Tensor
from tokenizers.implementations import ByteLevelBPETokenizer

from language_modeling.domain.modeling.utils.decoder import (
    BATCH_SIZE_TO_GENERATE_HIDDEN_STATES,
)
from language_modeling.domain.modeling.utils.data import SOS_TOKEN
from language_modeling.domain.modeling import DEVICE
from language_modeling.domain.modeling.model.neural_network.nn import LSTMModel
from language_modeling.domain.modeling.utils.trainer.trainer import TrainerUtils


class GreedyDecoder(TrainerUtils):
    def __init__(self, tokenizer: ByteLevelBPETokenizer, model: LSTMModel):
        self._tokenizer = tokenizer
        self._model = model
        self._put_model_on_the_device(self._model)
        self._put_model_to_eval_mode(self._model)

    @staticmethod
    def _wrap_list_of_ids_into_a_torch_tensor(sequence_of_ids: List[int]) -> Tensor:
        return tensor([sequence_of_ids], dtype=torch.long, device=DEVICE)

    @staticmethod
    def _add_sos_str_to_the_seed_str(seed_str: str) -> str:
        return "{} {}".format(SOS_TOKEN, seed_str)

    def _tokenize_text(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    @staticmethod
    def _get_top_k_word_ids(
        model_predictions, top_number_of_words_to_keep: int
    ) -> Tensor:
        _, top_k_words_ids = torch.topk(model_predictions, top_number_of_words_to_keep)
        return top_k_words_ids

    @staticmethod
    def _get_random_word_id_from_a_list_of_ids(list_of_ids: List[int]) -> int:
        return random.choice(list_of_ids)

    @staticmethod
    def _convert_tensor_to_list(tensor_of_ids: Tensor) -> List[int]:
        return tensor_of_ids.tolist()

    def generate_text(
        self,
        str_seed: str,
        maximum_sequence_length_to_generate: int,
        top_number_of_words_to_keep: int,
    ) -> str:
        str_seed = self._add_sos_str_to_the_seed_str(str_seed)
        model_input = self._tokenize_text(str_seed)
        model_input = self._wrap_list_of_ids_into_a_torch_tensor(model_input)
        hidden_states = self._model.init_hidden_states(
            BATCH_SIZE_TO_GENERATE_HIDDEN_STATES
        )

        ids_to_decode = [model_input]

        with torch.no_grad():
            for _ in range(maximum_sequence_length_to_generate):
                model_predictions, hidden_states = self._get_model_output(
                    self._model, model_input, hidden_states
                )
                model_predictions = self._squeeze_tensor(model_predictions)
                top_k_words_ids = self._get_top_k_word_ids(
                    model_predictions, top_number_of_words_to_keep
                )
                top_k_words_ids_list = self._convert_tensor_to_list(top_k_words_ids)
                model_input = self._get_random_word_id_from_a_list_of_ids(
                    top_k_words_ids_list
                )
                ids_to_decode.append(model_input)
                model_input = self._wrap_list_of_ids_into_a_torch_tensor([model_input])

        return self._tokenizer.decode(ids_to_decode, skip_special_tokens=True)
