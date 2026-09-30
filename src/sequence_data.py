"""Turn cleaned articles into padded word-id batches, and load fastText vectors for them.

Shared by the word-level neural models (BiLSTM, TextCNN). Expects the clean_text column
produced by preprocess.clean_splits.
"""

import gzip
import hashlib
import urllib.request
from collections import Counter

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import EMBEDDINGS_DIR, FASTTEXT_URL, SEED, make_output_dirs

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


def build_vocab(texts, min_count=3, max_size=None):
    """Map words to ids, most frequent first. Build it on the training split only."""
    counts = Counter(word for text in texts for word in text.split())
    words = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
    if max_size:
        words = words[: max_size - 2]
    return {word: i for i, word in enumerate([PAD, UNK] + words)}


def encode(text, vocab, max_len):
    """Word ids for the first max_len words. The head of an article carries most of the topic."""
    ids = [vocab.get(word, UNK_ID) for word in text.split()[:max_len]]
    return ids or [UNK_ID]


class ArticleDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.ids = [torch.tensor(encode(t, vocab, max_len), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.ids[index], self.labels[index]


def collate(batch, min_len=1):
    """Pad a batch to its longest article. min_len lets convolution models ask for a floor."""
    ids, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids])
    padded = pad_sequence(ids, batch_first=True, padding_value=PAD_ID)
    if padded.size(1) < min_len:
        padded = torch.nn.functional.pad(padded, (0, min_len - padded.size(1)), value=PAD_ID)
    return padded, lengths, torch.stack(labels)


def make_loader(frame, vocab, max_len, batch_size, shuffle=False, min_len=1):
    dataset = ArticleDataset(frame["clean_text"], frame["label"], vocab, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate(batch, min_len),
        generator=torch.Generator().manual_seed(SEED),
    )


def download_fasttext():
    make_output_dirs()
    path = EMBEDDINGS_DIR / "cc.sw.300.vec.gz"
    if not path.exists():
        print("Downloading fastText Swahili vectors (large file, only needed once)...")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(FASTTEXT_URL, partial)
        partial.rename(path)
    return path


def load_fasttext(vocab, dim=300):
    """Embedding matrix aligned with vocab, and the share of vocab words found in fastText.

    fastText lists words from most to least frequent, so the first lowercase match wins.
    Missing words get random vectors on the same scale. The matrix is cached per vocabulary.
    """
    key = hashlib.md5(" ".join(vocab).encode()).hexdigest()[:10]
    cache = EMBEDDINGS_DIR / f"fasttext_{key}.npz"
    if cache.exists():
        saved = np.load(cache)
        return saved["matrix"], float(saved["coverage"])

    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    found = np.zeros(len(vocab), dtype=bool)
    with gzip.open(download_fasttext(), "rt", encoding="utf-8", errors="ignore") as vectors:
        next(vectors)
        for line in vectors:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            index = vocab.get(parts[0].lower())
            if index is not None and not found[index]:
                matrix[index] = np.array(parts[1:], dtype=np.float32)
                found[index] = True

    coverage = float(found[2:].mean())
    scale = matrix[found].std() if found.any() else 0.1
    missing = ~found
    rng = np.random.default_rng(SEED)
    matrix[missing] = rng.normal(0, scale, size=(missing.sum(), dim))
    matrix[PAD_ID] = 0

    np.savez(cache, matrix=matrix, coverage=coverage)
    return matrix, coverage
