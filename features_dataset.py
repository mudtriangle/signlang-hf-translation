import torch
import os
import numpy as np


class TranslationFeatures(torch.utils.data.Dataset):
    def __init__(self, paths_file, sentences_file, tokenizer, cache_root, skiplines=0):
        with open(paths_file, "r") as f:
            paths = f.readlines()
        self.paths = []
        for p in paths:
            if p.strip() == "/" or p.strip() == "":
                continue
            
            p = p.split("\t")[0]
            p = os.path.basename(p)
            p = os.path.join(cache_root, f"{p}.npy")
            self.paths.append(p)

        with open(sentences_file, "r") as f:
            sentences = f.readlines()
        self.sentences = [s.strip() for s in sentences]
        self.sentences = tokenizer.batch_encode_plus(self.sentences).input_ids

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx])
        if len(arr) > 256:
            arr = arr[:256]
        
        return {
            "input_features": torch.tensor(arr),
            "labels": torch.tensor(self.sentences[idx]),
        }

