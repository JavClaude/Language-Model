import os
import pickle

from language_modeling.domain.modeling.utils.saver import WRITE_BYTE, ARTEFACT_NAME
from language_modeling.domain.modeling.utils.data.dataset import LanguageModelingDataset
from language_modeling.domain.modeling.model.neural_network.nn import LSTMModel


class Saver:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir

    def save_preprocessor_and_model(
        self, dataset: LanguageModelingDataset, model: LSTMModel
    ) -> None:
        tuple_of_preprocessor_and_model = tuple((dataset, model))
        joined_path = self._get_joined_path(self.log_dir, ARTEFACT_NAME)
        with open(joined_path, WRITE_BYTE) as file:
            pickle.dump(tuple_of_preprocessor_and_model, file)

    @staticmethod
    def _get_joined_path(first_path: str, second_path: str) -> str:
        return os.path.join(first_path, second_path)
