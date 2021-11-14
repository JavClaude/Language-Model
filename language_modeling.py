import argparse
import random
from typing import Any


import numpy as np
from tokenizers import Tokenizer
import torch
from torch import load, long, manual_seed, no_grad, save, tensor, topk
from torch.cuda import is_available
from torch.nn import Softmax

from language_model.model import LstmModel
from language_model.preprocessing import get_bpe_tokenizer, LanguageModelingDataset

device = "cuda" if is_available else "cpu"

random.seed(32)
np.random.seed(32)
manual_seed(32)

def train(args: Any) -> None:
    tokenizer = get_bpe_tokenizer()

    train_data_iterator = LanguageModelingDataset(args.batch_size, args.bptt, True)
    train_data_iterator.fit(args.train_data, tokenizer, args.vocabulary_size)

    args.vocabulary_size = tokenizer.get_vocab_size()

    model = LstmModel(**vars(args))
    
    if args.eval_data is not None:
        eval_data_iterator = LanguageModelingDataset(args.batch_size, args.bptt, False)
        eval_data_iterator.fit(args.eval_data, tokenizer, args.vocabulary_size)
        model.fit(
            train_data_iterator,
            eval_data_iterator,
            args.n_epochs,
            lr=args.lr,
            optimizer_name=args.optimizer_name
        )
    else:
        model.fit(
            train_data_iterator, 
            args.n_epochs,
            lr=args.lr,
            optimizer_name=args.optimizer_name
        )

    path_to_save_artifacts = model.writer.get_logdir()
    tokenizer.save(path_to_save_artifacts + "/tokenizer.json")
    model.to("cpu")
    save(model, path_to_save_artifacts + "/model.pt")

def predict(args: Any) -> None:
    sequence_seed = args.sequence_seed
    model = load(args.model_weights, map_location=torch.device(device))
    softmax_function = Softmax(0)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    model.eval()
    model.to(device)

    hidden_states = model.init_hidden(1)

    input_tokens = tokenizer.encode("<SOS> " + sequence_seed).ids

    for _ in range(args.maximum_sequence_length):
        with no_grad():
            
            input_tensor = tensor([input_tokens], dtype=long, device=device)

            logits, hidden_states = model(
                (
                input_tensor,
                hidden_states
                )
            )

            _, top_tokens = topk(
                softmax_function(
                    logits[0][-1]
                ),
                2
            )

            random_token = random.choice(top_tokens.tolist())
            
            input_tokens.append(random_token)

    print(
        tokenizer.decode(input_tokens, skip_special_tokens=True)
    )

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
    cmd1_parser.add_argument("--batch_size", type=int, default=64, required=False, help="batch size used for gradient descent")
    cmd1_parser.add_argument("--bptt", type=int, default=64, required=False, help="maximum sequence length to used for gradient descent")
    cmd1_parser.add_argument("--n_epochs", type=int, default=3, required=False, help="maximum number of epochs used to train the model")
    cmd1_parser.add_argument("--lr", type=float, default=0.00001, required=False, help="learning rate used to train the model")
    cmd1_parser.add_argument("--optimizer_name", type=str, default="Adam", required=False, help="Optimizer used to train the model: Adam or SGD")

    cmd2_parser = commands.add_parser("predict", help="Predict sequence from a pre trained language model")
    cmd2_parser.add_argument("--model_weights", type=str, default=False, required=True, help="path to the pre trained model weights")
    cmd2_parser.add_argument("--tokenizer", type=str, default=False, required=True, help="path to the pre trained tokenizer")
    cmd2_parser.add_argument("--sequence_seed", type=str, default=False, required=True, help="sequence seed to generate text from")
    cmd2_parser.add_argument("--maximum_sequence_length", type=int, default=128, required=False, help="maximum number of tokens to generate")

    args = parser.parse_args()

    if args.subcommand == "train":
        train(args)
    elif args.subcommand == "predict":
        predict(args)