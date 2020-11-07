import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_iterator, criterion, optimizer, global_train_it):
    pass

def eval_model(model, test_iterator, criterion, global_eval_it):
    pass