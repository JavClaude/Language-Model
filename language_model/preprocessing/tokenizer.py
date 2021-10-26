from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers import pre_tokenizers
from tokenizers.processors import ByteLevel

def get_bpe_tokenizer() -> Tokenizer:
    """Return an instance of Tokenizer

    get_tokenizer is a function that instantiate a BPE Tokenizer object.

    Returns
    -------
    tokenizer: Tokenizer
        Instance of Tokenizer object
    """
    tokenizer = Tokenizer(
        BPE(
            continuing_subword_prefix="##",
            unk_token="<UNK>",
            end_of_word_suffix="</w>"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = BPEDecoder()
    tokenizer.post_processor = ByteLevel()
    tokenizer.add_special_tokens(
        ["<SOS>", "<PAD>", "<EOS>"]
    )
    return tokenizer