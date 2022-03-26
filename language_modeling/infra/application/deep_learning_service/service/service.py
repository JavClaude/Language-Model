from fastapi import FastAPI
from tokenizers.implementations import ByteLevelBPETokenizer

from language_modeling.domain.modeling.model.neural_network.nn import LSTMModel
from language_modeling.domain.modeling.utils.decoder.greedy_decoder import GreedyDecoder
from language_modeling.infra.application.deep_learning_service.service import GENERATE_TEXT_ROOT_PATH
from language_modeling.infra.application.deep_learning_service.data_model.data_model import TextInput, TextOutput


class DeepLearningService:
    def __init__(self, tokenizer: ByteLevelBPETokenizer, model: LSTMModel) -> None:
        self._decoder = GreedyDecoder(tokenizer, model)
        self._api = FastAPI()

    def get_api(self) -> FastAPI:
        return self._api

    def _add_generate_text_endpoint(self) -> None:
        @self._api.post(GENERATE_TEXT_ROOT_PATH, response_model=TextOutput)
        def generate_text(text_inputs: TextInput) -> TextOutput:
            generated_text = self._decoder.generate_text(
                text_inputs.seed_str,
                text_inputs.maximum_sequence_length,
                text_inputs.top_k_word,
            )
            return TextOutput(
                seed_str=text_inputs.seed_str, generated_text=generated_text
            )

    def build_api(self) -> None:
        self._add_generate_text_endpoint()
