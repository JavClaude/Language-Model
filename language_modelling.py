import argparse
import pickle

from torch.cuda import is_available

from language_model.model import LstmModel
from language_model.preprocessing import LanguageModelingDataset, get_bpe_tokenizer

device = "cuda" if is_available else "cpu"

def train(args):
    tokenizer = get_bpe_tokenizer()
    model = LstmModel(**vars(args))

    train_data_iterator = LanguageModelingDataset(args.batch_size, args.bptt, True)
    train_data_iterator.fit(args.train_data, tokenizer, args.vocabulary_size)
    
    if args.eval_data is not None:
        eval_data_iterator = LanguageModelingDataset(args.batch_size, args.bptt, False)
        eval_data_iterator.fit(args.eval_data, tokenizer, args.vocabulary_size)
        model.fit(
            train_data_iterator,
            eval_data_iterator,
            args.n_epochs
        )
    else:
        model.fit(train_data_iterator, args.n_epochs)

def predict(args):
    print("predict")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or load a lstm based language model")
    commands = parser.add_subparsers(dest="subcommand")

    cmd1_parser = commands.add_parser("train", help="Train a language model")
    cmd1_parser.add_argument("--train_data", type=str, required=True, help="path to the train txt file")
    cmd1_parser.add_argument("--eval_data", type=str, required=False, help="path to the eval txt file")
    cmd1_parser.add_argument("--vocabulary_size", type=int, default=30000, required=False, help="desired vocabulary size for your language model")
    cmd1_parser.add_argument("--embedding_dimension", type=int, default=300, required=False, help="desired embedding dimension for your language model")
    cmd1_parser.add_argument("--dropout_rnn", type=float, default=0.4, required=False, help="desired dropout for the lstm layer of your language model")
    cmd1_parser.add_argument("--hidden_units", type=int, default=256, required=False, help="desired hidden lstm units for your language model")
    cmd1_parser.add_argument("--num_layers", type=int, default=2, required=False, help="desired number of lstm layer for your language model")
    cmd1_parser.add_argument("--batch_size", type=int, default=128, required=False, help="batch size used for gradient descent")
    cmd1_parser.add_argument("--bptt", type=int, default=128, required=False, help="maximum sequence length to used for gradient descent")
    cmd1_parser.add_argument("--n_epochs", type=int, default=3, required=False, help="maximum number of epochs used to train the model")

    cmd2_parser = commands.add_parser("predict", help="Predict sequence from a pre trained language model")
    cmd2_parser.add_argument("--model_weights", type=str, default=False, required=True, help="path to the pre trained model weights")
    cmd2_parser.add_argument("--tokenizer", type=str, default=False, required=True, help="path to the pre trained tokenizer")
    cmd2_parser.add_argument("--sequence_seed", type=str, default=False, required=True, help="sequence seed to generate text from")

    args = parser.parse_args()

    if args.subcommand == "train":
        train(args)
    elif args.subcommand == "predict":
        predict(args)