import os
import argparse
import tokenizers

def train_trokenizer(**kwargs):
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train(kwargs.get("path_to_textfile"), vocab_size=kwargs.get("num_merges"))
    tokenizer.add_special_tokens(["<SOS>", "<PAD>", "<EOS>"])

    if __name__ == "__main__":
        if not os.path.isdir(".tmp"):
            os.mkdir(".tmp")
        else:
            pass
        tokenizer.save(".tmp/tokenizer.bin")
    else:
        return tokenizer
    

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_textfile", type=str, required=True)
    argument_parser.add_argument("--num_merges", type=int, required=False, default=30000)

    args = argument_parser.parse_args()

    train_trokenizer(**vars(args))
