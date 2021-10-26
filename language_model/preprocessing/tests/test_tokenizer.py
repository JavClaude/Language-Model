from tokenizers import Tokenizer

from language_model.preprocessing import get_bpe_tokenizer


def test_get_bpe_tokenizer() -> None:
    tokenizer = get_bpe_tokenizer()
    assert isinstance(tokenizer, Tokenizer)