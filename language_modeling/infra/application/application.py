from typing import Tuple
import pickle

from fastapi import FastAPI

from language_modeling.infra.data.data_model import TextInput, TextOutput
from language_modeling.domain.modeling.utils.data.dataset import LanguageModelingDataset
from language_modeling.domain.modeling.model.neural_network.nn import LSTMModel
from language_modeling.domain.modeling.utils.decoder.greedy_decoder import GreedyDecoder


class APIBuilder:
    def __init__(self, path_to_preprocessor_and_model: str) -> None:
        self._api = self._create_app()
        preprocessor_and_model = self._load_preprocessor_and_model(
            path_to_preprocessor_and_model
        )
        self._decoder = GreedyDecoder(
            preprocessor_and_model[0]._tokenizer, preprocessor_and_model[1]
        )

    @staticmethod
    def _create_app() -> FastAPI:
        return FastAPI()

    def _add_generate_text_root(self) -> None:
        @self._api.post("/v1/generate_text", response_model=TextOutput)
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
        self._add_generate_text_root()

    def get_api(self) -> FastAPI:
        return self._api

    @staticmethod
    def _load_preprocessor_and_model(
        path_to_preprocessor_and_model: str,
    ) -> Tuple[LanguageModelingDataset, LSTMModel]:
        with open(path_to_preprocessor_and_model, "rb") as file:
            preprocessor_and_model = pickle.load(file)
        return preprocessor_and_model


if __name__ == "__main__":
    api_builder = APIBuilder()
    api_builder.build_api()

    import uvicorn

    uvicorn.run(api_builder.get_api(), host="0.0.0.0", port=5055)
