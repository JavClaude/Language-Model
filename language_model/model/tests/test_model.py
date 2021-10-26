from typing import Tuple
import pytest
from torch import randint, tensor, zeros
from torch.cuda import is_available

from language_model.model import LstmModel
from language_model.preprocessing import LanguageModelingDataset, get_bpe_tokenizer

device = "cuda" if is_available() else "cpu"

BATCH_SIZE = 15
BPTT = 4
EMBEDDING_DIMENSION = 20
EPOCHS = 3
DROPOUT_RNN = 0.3
HIDDEN_UNITS = 10
NUM_LAYERS = 2
VOCABULARY_SIZE = 300

PATH_TO_TRAIN_TEXTS = "/tmp/training_texts.txt"
PATH_TO_EVAL_TEXTS = "/tmp/testing_texts.txt"

@pytest.fixture
def model(
    vocabulary_size: int = VOCABULARY_SIZE,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    hidden_units: int = HIDDEN_UNITS,
    num_layers: int = NUM_LAYERS,
    dropout_rnn: float = DROPOUT_RNN
) -> LstmModel:
    return LstmModel(
        vocabulary_size,
        embedding_dimension, 
        hidden_units, 
        num_layers,
        dropout_rnn
    )

@pytest.fixture
def mock_data(
    batch_size: int = BATCH_SIZE,
    bptt: int = BPTT,
    vocabulary_size: int = VOCABULARY_SIZE,
) -> tensor:
    return randint(0, vocabulary_size, size=(batch_size, bptt), device=device)

@pytest.fixture
def data_iterators(
    batch_size: int = BATCH_SIZE,
    bptt: int = BPTT 
) -> Tuple[LanguageModelingDataset, LanguageModelingDataset]:
    texts_for_train = ["This is a text for a test"] * 100

    with open(PATH_TO_TRAIN_TEXTS, "w") as file:
        file.writelines(texts_for_train)

    texts_for_eval = ["This is anoter text for a test"] * 50

    with open(PATH_TO_EVAL_TEXTS, "w") as file:
        file.writelines(texts_for_eval)

    return (
        LanguageModelingDataset(batch_size, bptt, True),
        LanguageModelingDataset(batch_size, bptt, False)
    )

def test_init_hidden(model: LstmModel, batch_size: int = BATCH_SIZE) -> None:
    hidden_states = model.init_hidden(batch_size)
    assert hidden_states[0].shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_UNITS), "Bad shape return for init hidden :("

def test_init_memory(model: LstmModel, batch_size: int = BATCH_SIZE) -> None:
    hidden_states = model.init_hidden(batch_size)
    assert hidden_states[1].shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_UNITS), "Bad shape return for init hidden :("

def test_forward_pass(model: LstmModel, mock_data: tensor, batch_size: int = BATCH_SIZE, ) -> None:
    model.to(device)
    hiddens = model.init_hidden(batch_size)
    logits, _ = model((mock_data, hiddens))
    assert logits.shape == (BATCH_SIZE, BPTT, VOCABULARY_SIZE)

def test_fit(model: LstmModel, data_iterators: Tuple[LanguageModelingDataset, LanguageModelingDataset]) -> None:
    tokenizer = get_bpe_tokenizer()

    train_data_iterator = data_iterators[0]
    train_data_iterator.fit(PATH_TO_TRAIN_TEXTS, tokenizer, VOCABULARY_SIZE)

    eval_data_iterator = data_iterators[1]
    eval_data_iterator.fit(PATH_TO_EVAL_TEXTS, tokenizer, VOCABULARY_SIZE)

    model.fit(
        train_data_iterator,
        eval_data_iterator, 
        epochs=EPOCHS
    )