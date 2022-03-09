import argparse

from tokenizers.implementations import ByteLevelBPETokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from language_modeling.domain.modeling.utils.data.dataset import LanguageModelingDataset
from language_modeling.domain.modeling.utils.data.dataloader import (
    LanguageModelingDataLoader,
)
from language_modeling.domain.modeling.model.neural_network.nn import LSTMModel
from language_modeling.domain.modeling.utils.trainer.trainer import Trainer
from language_modeling.domain.modeling.utils.saver.saver import Saver
from language_modeling.domain.modeling.utils.logger.tensorboard_logger import (
    TensorboardLogger,
)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_train_data", type=str, required=True)
    argument_parser.add_argument(
        "--path_to_eval_data", type=str, required=False, default=None
    )
    argument_parser.add_argument(
        "--path_to_save_preprocessor_and_model", type=str, required=False, default="."
    )
    argument_parser.add_argument("--n_epochs", type=int, required=False, default=3)
    argument_parser.add_argument("--batch_size", type=int, required=False, default=32)
    argument_parser.add_argument("--bptt", type=int, required=False, default=64)
    argument_parser.add_argument("--lr", type=float, required=False, default=0.0001)
    argument_parser.add_argument(
        "--vocabulary_size", type=int, required=False, default=20000
    )
    argument_parser.add_argument(
        "--embedding_dimension", type=int, required=False, default=300
    )
    argument_parser.add_argument(
        "--hidden_units_for_lstm", type=int, required=False, default=256
    )
    argument_parser.add_argument(
        "--num_of_lstm_layer", type=int, required=False, default=1
    )
    argument_parser.add_argument(
        "--n_decoder_blocks", type=int, required=False, default=5
    )

    arguments = argument_parser.parse_args()

    train_language_modeling_dataset = LanguageModelingDataset(
        arguments.batch_size, arguments.bptt
    )
    train_language_modeling_dataset.set_tokenizer(ByteLevelBPETokenizer())
    train_language_modeling_dataset.fit(
        arguments.path_to_train_data, vocabulary_size=arguments.vocabulary_size
    )

    train_language_modeling_dataloader = LanguageModelingDataLoader(
        arguments.bptt,
        train_language_modeling_dataset.transform(
            arguments.path_to_train_data, return_target=True
        ),
    )

    model = LSTMModel(
        arguments.vocabulary_size,
        arguments.embedding_dimension,
        arguments.hidden_units_for_lstm,
        arguments.n_decoder_blocks,
        arguments.num_of_lstm_layer,
    )

    trainer = Trainer(arguments.batch_size)
    trainer.set_logger(TensorboardLogger())

    if arguments.path_to_eval_data:
        eval_language_modeling_dataloader = LanguageModelingDataLoader(
            arguments.bptt,
            train_language_modeling_dataset.transform(
                arguments.path_to_eval_data, return_target=True
            ),
        )

        trainer.train(
            model,
            train_language_modeling_dataloader,
            CrossEntropyLoss(),
            Adam(model.parameters()),
            eval_language_modeling_dataloader,
            2,
        )

    else:
        trainer.train(
            model,
            train_language_modeling_dataloader,
            CrossEntropyLoss(),
            Adam(model.parameters(), arguments.lr),
            None,
            2,
        )

    saver = Saver(arguments.path_to_save_preprocessor_and_model)
    saver.save_preprocessor_and_model(train_language_modeling_dataset, model)


if __name__ == "__main__":
    main()
