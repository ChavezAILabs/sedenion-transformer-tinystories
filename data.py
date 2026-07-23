"""data.py — char-level Tiny Shakespeare loader for the ZDA smoke run.

Downloads the corpus once into data/tinyshakespeare.txt, builds a char
vocab, and serves seeded random batches so data order is reproducible
and identical across variants for a given seed (spec section 4).
"""

import os
import urllib.request

import numpy as np
import torch

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_PATH = os.path.join(DATA_DIR, "tinyshakespeare.txt")


def ensure_text() -> str:
    if not os.path.exists(RAW_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading Tiny Shakespeare -> {RAW_PATH}")
        urllib.request.urlretrieve(URL, RAW_PATH)
    with open(RAW_PATH, encoding="utf-8") as f:
        return f.read()


class CharDataset:
    def __init__(self, split_frac: float = 0.9):
        text = ensure_text()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        ids = np.array([self.stoi[c] for c in text], dtype=np.int64)
        n = int(len(ids) * split_frac)
        self.train_ids = torch.from_numpy(ids[:n])
        self.val_ids = torch.from_numpy(ids[n:])

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split: str, batch_size: int, ctx: int,
                  generator: torch.Generator):
        """Random contiguous chunks; all randomness comes from `generator`,
        so two runs with identically seeded generators see identical data."""
        data = self.train_ids if split == "train" else self.val_ids
        ix = torch.randint(len(data) - ctx - 1, (batch_size,),
                           generator=generator)
        x = torch.stack([data[i:i + ctx] for i in ix])
        y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
        return x, y


if __name__ == "__main__":
    ds = CharDataset()
    print(f"vocab_size={ds.vocab_size}, "
          f"train={len(ds.train_ids)} chars, val={len(ds.val_ids)} chars")
    g = torch.Generator().manual_seed(0)
    x, y = ds.get_batch("train", 2, 32, g)
    print("sample:", repr(ds.decode(x[0].tolist())))
