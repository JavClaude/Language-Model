import argparse

from tokenizers.implementations import ByteLevelBPETokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from language_model.domain.modeling.data.dataset import LanguageModelingDataset
from language_model.domain.modeling.data.dataloader import LanguageModelingDataLoader
from language_model.domain.modeling.model.neural_network.nn import LSTMModel
from language_model.domain.modeling.model.utils.training import Trainer
from language_model.domain.modeling.model.utils.logger.tensorboard_logger import TensorboardLogger


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_train_data", type=str, required=True)
    argument_parser.add_argument("--path_to_eval_data", type=str, required=False, default=None)
    argument_parser.add_argument("--batch_size", type=int, required=False, default=32)
    argument_parser.add_argument("--bptt", type=int, required=False, default=64)
    argument_parser.add_argument("--vocabulary_size", type=int, required=False, default=20000)
    argument_parser.add_argument("--embedding_dimension", type=int, required=False, default=300)
    argument_parser.add_argument("--hidden_units_for_lstm", type=int, required=False, default=256)
    argument_parser.add_argument("--num_of_lstm_layer", type=int, required=False, default=1)
    argument_parser.add_argument("--n_decoder_blocks", type=int, required=False, default=5)

    arguments = argument_parser.parse_args()

    train_language_modeling_dataset = LanguageModelingDataset(arguments.batch_size, arguments.bptt)
    train_language_modeling_dataset.set_tokenizer(ByteLevelBPETokenizer())
    train_language_modeling_dataset.fit(arguments.path_to_train_data, vocabulary_size=arguments.vocabulary_size)

    train_language_modeling_dataloader = LanguageModelingDataLoader(
        arguments.bptt, train_language_modeling_dataset.transform(
            arguments.path_to_train_data, return_target=True
        )
    )

    model = LSTMModel(
        arguments.vocabulary_size,
        arguments.embedding_dimension,
        arguments.hidden_units_for_lstm,
        arguments.n_decoder_blocks,
        arguments.num_of_lstm_layer
    )

    trainer = Trainer(arguments.batch_size)
    trainer.set_logger(TensorboardLogger())

    if arguments.path_to_eval_data:
        eval_language_modeling_dataloader = LanguageModelingDataLoader(
            arguments.bptt, train_language_modeling_dataset.transform(
                arguments.path_to_eval_data, return_target=True
            )
        )

        trainer.train(
            model,
            train_language_modeling_dataloader,
            CrossEntropyLoss(),
            Adam(model.parameters()),
            eval_language_modeling_dataloader,
            2
        )

    else:
        trainer.train(
            model,
            train_language_modeling_dataloader,
            CrossEntropyLoss(),
            Adam(model.parameters()),
            None,
            2
        )


if __name__ == "__main__":
    main()
