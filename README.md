# LSTM Language-Model

In this repository you will find all elements to train a subword level LSTM Language Model.

First, create a virtualenv environment and install all dependencies
```sh
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then, prepare two line-by-line files for training / testing, start a tensorboard container and start training
```sh
docker image build --tag tensorboard_lm tensorboard/
docker container run --restart always -d -p 5001:5001 tensorboard_lm

python language_modeling.py train --train_data train_texts.txt --eval_data eval_texts.txt --lr 0.0001 --n_epochs=3 --hidden_units 512 
```