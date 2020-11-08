import os
import json
import logging
import argparse

import tqdm
import torch
import mlflow
import tokenizers

from Model.model import LSTMModel
from Tokenizer.train_tokenizer import train_trokenizer
from Training.training_eval import train_model, eval_model
from Utils.data import TextDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

## Mlflow configuration ##
#
# mlflow.set_experiment("") # Experiment name
#
# os.environ['MLFLOW_TRACKING_URI'] # Mlflow tracking server for example http://localhost:5000
# os.environ['MLFLOW_S3_ENDPOINT_URL'] # S3 Registry (container minIO) for example http://localhost:9000
#
##########################

def main(**kwargs) -> None:

    if kwargs.get("path_to_tokenizer") is None:
        tokenizer = train_trokenizer(**{
            "path_to_textfile": kwargs.get("path_to_data_train"),
            "num_merges": kwargs.get("num_merges")
        })
    else:
        tokenizer = tokenizers.Tokenizer.from_file(kwargs.get("path_to_tokenizer"))

    kwargs["vocab_size"] = tokenizer.get_vocab_size()

    trainDataset = TextDataset(kwargs.get("path_to_data_train"), tokenizer, kwargs.get("bptt"), kwargs.get("batch_size"))
    testDataset = TextDataset(kwargs.get("path_to_data_test"), tokenizer, kwargs.get("bptt"), kwargs.get("batch_size"))

    Model = LSTMModel(**kwargs)
    Criterion = torch.nn.CrossEntropyLoss()
    Optimizer = torch.optim.Adam(Model.parameters(), lr=kwargs.get("lr"))

    # Just in case, clean gradient, put the model on the GPU (if available)
    Model.zero_grad()
    Model.to(device)

    logger.info("Start training for: {}".format(kwargs.get("epochs")))

    global_train_it = 0
    global_eval_it = 0

    best_loss = 50

    for _ in tqdm.tqdm(range(kwargs.get("epochs"))):
        # Train
        _, global_train_it = train_model(
            Model,
            trainDataset,
            Criterion,
            Optimizer,
            global_train_it
        )

        # Test
        eval_loss, global_eval_it = eval_model(
            Model,
            testDataset,
            Criterion,
            global_eval_it
        )

        if eval_loss < best_loss:
            # Store artifacts
            with open("config_file.json", "w") as file:
                json.dumps(kwargs, file)
            
            torch.save(Model.state_dict(), "model_state_dict.pt")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_data_train", type=str, required=True)
    argument_parser.add_argument("--path_to_data_test", type=str, required=True)
    argument_parser.add_argument("--path_to_tokenizer", type=str, required=False)
    argument_parser.add_argument("--num_merges", type=int, required=False, default=30000)
    argument_parser.add_argument("--epochs", type=int, required=False, default=3)
    argument_parser.add_argument("--batch_size", type=int, required=False, default=64)
    argument_parser.add_argument("--bptt", type=int, required=False, default=128)
    argument_parser.add_argument("--lr", type=float, required=False, default=0.0001)
    argument_parser.add_argument("--clip_grad_norm", type=float, required=False, default=3)
    argument_parser.add_argument("--embedding_dim", type=int, required=False, default=300)
    argument_parser.add_argument("--hidden_units", type=int, required=False, default=256)
    argument_parser.add_argument("--n_layers", type=int, required=False, default=3)
    argument_parser.add_argument("--bidirectional", type=bool, required=False, default=False)
    argument_parser.add_argument("--dropout_rnn", type=float, required=False, default=0.4)
    argument_parser.add_argument("--dropout", type=float, required=False, default=0.5)

    arguments = argument_parser.parse_args()

    main(**vars(arguments))

