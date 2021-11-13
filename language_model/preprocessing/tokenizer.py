from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

def get_bpe_tokenizer() -> Tokenizer:
    """Return an instance of Tokenizer

    get_tokenizer is a function that instantiate a BPE Tokenizer object.

    Returns
    -------
    tokenizer: Tokenizer
        Instance of Tokenizer object
    """
    tokenizer = Tokenizer(
        BPE()
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.add_special_tokens(
        ["<SOS>", "<PAD>", "<EOS>"]
    )
    return tokenizer