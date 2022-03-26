import pickle
from typing import Tuple

from language_modeling.infra.application.deep_learning_service.utils import READ_BYTES
from language_modeling.domain.modeling.utils.data.dataset import LanguageModelingDataset
from language_modeling.domain.modeling.model.neural_network.nn import LSTMModel


class ArtifactsLoader:
    @staticmethod
    def load_preprocessor_and_model(path_to_preprocessor_and_model_object: str) -> Tuple[LanguageModelingDataset, LSTMModel]:
        with open(path_to_preprocessor_and_model_object, READ_BYTES) as file:
            preprocessor_and_model_object = pickle.load(file)
        return preprocessor_and_model_object
