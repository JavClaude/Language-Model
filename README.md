# language-model

A small package to train a sub word language model based on LSTM neural network

## Installation

To install the package you can install it using the following command
```bash
make Makefile
```

## Train

After, you can run a training using the following command
```bash
train_language_model 
  --path_to_train_data <your_path_to_train_data_here> 
  --path_to_eval_data <your_path_to_eval_data_here>
  --n_epochs 3 
  --batch_size 256 
  --n_decoder_blocks 10
```

## Logging

Note that a tensorboard logger is used by the entrypoint, you can start a tensorboard server using the `Dockerfile` located at `language_modeling/infra/tensorboard`:
```bash
docker image build --tag tensorboard_lm:latest # build the docker image
docker container run --rm --restart always -d -p 5001:5001 tensorboard_lm:latest 
```
