from os import path
import pytest

from language_model.model.tests.test_model import BATCH_SIZE, VOCABULARY_SIZE
from language_model.preprocessing import LanguageModelingDataset, get_bpe_tokenizer

BATCH_SIZE = 12
BPTT = 4
ITERATOR_LENGTH = 108

PATH_TO_TRAIN_TEXTS = "/tmp/training_texts.txt"
PATH_TO_EVAL_TEXTS = "/tmp/testing_texts.txt"

texts_for_train = ["This is a text for a test"] * 100

with open(PATH_TO_TRAIN_TEXTS, "w") as file:
    file.writelines(texts_for_train)

texts_for_eval = ["This is anoter text for a test"] * 50

with open(PATH_TO_EVAL_TEXTS, "w") as file:
    file.writelines(texts_for_eval)

def test_iterator_length():
    tokenizer = get_bpe_tokenizer()
    train_iterator = LanguageModelingDataset(BATCH_SIZE, BPTT, True)
    train_iterator.fit(
        PATH_TO_TRAIN_TEXTS,
        tokenizer,
        VOCABULARY_SIZE
    )
    assert len(train_iterator) == ITERATOR_LENGTH

def test_total_number_of_batches():
    tokenizer = get_bpe_tokenizer()
    train_iterator = LanguageModelingDataset(BATCH_SIZE, BPTT, True)
    train_iterator.fit(
        PATH_TO_TRAIN_TEXTS,
        tokenizer,
        VOCABULARY_SIZE
    )
    assert train_iterator.total_number_of_batches == 27