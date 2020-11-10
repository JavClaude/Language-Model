import os
import argparse
import tokenizers

def train_trokenizer(**kwargs):

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
    tokenizer.decoder = tokenizers.decoders.BPEDecoder(suffix = "</w>")
    tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=True)
    tokenizer.add_special_tokens(["<SOS>", "<PAD>", "<EOS>"])

    trainer = tokenizers.trainers.BpeTrainer(vocab_size=kwargs.get("num_merges"), continuing_subword_prefix="##", end_of_word_suffix="</w>")
    tokenizer.train(trainer, [kwargs.get("path_to_textfile")])

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
