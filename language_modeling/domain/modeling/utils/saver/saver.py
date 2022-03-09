import pickle


from language_modeling.domain.modeling.utils.data.dataset import LanguageModelingDataset
from language_modeling.domain.modeling.model.neural_network.nn import LSTMModel


class Saver:
    def __init__(self, path_to_save_data: str) -> None:
        self.path_to_save_data = path_to_save_data

    def save_preprocessor_and_model(
        self, dataset: LanguageModelingDataset, model: LSTMModel
    ) -> None:
        tuple_of_preprocessor_and_model = tuple((dataset, model))

        with open(self.path_to_save_data, "wb") as file:
            pickle.dump(tuple_of_preprocessor_and_model, file)
