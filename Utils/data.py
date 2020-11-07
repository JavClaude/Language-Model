import os
import itertools
import logging
import tqdm

import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TextDataset():
    def __init__(self,
                path_to_txt: str,
                tokenizer,
                bptt,
                batch_size: int) -> None:
        
        self.bptt = bptt

        texts_ids = []
        with open(path_to_txt, "r") as file:
            for text in tqdm.tqdm(file.readlines(), desc="Reading & tokenize file: {}".format(path_to_txt)):
                texts_ids.append(
                    tokenizer.encode("<SOS> " + text + " <EOS>").ids
                )
        
        flat_texts_ids = list(itertools.chain.from_iterable((texts_ids)))    
        self.n_batches = len(flat_texts_ids) // batch_size
        flat_texts_ids = np.array(flat_texts_ids[0: self.n_batches*batch_size])
        self.texts_ids = flat_texts_ids.reshape(batch_size, -1)
    
    def get_batches(self, i):
        return (
            torch.tensor(self.texts_ids[:, i+1:i+self.bptt+1], dtype=torch.long, device=device),
            torch.tensor(self.texts_ids[:, i+1:i+self.bptt+1], dtype=torch.long, device=device)
        )




        