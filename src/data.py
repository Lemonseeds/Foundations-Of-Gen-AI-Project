"""
Data loading, cleaning, tokenisation and autoregressive sequence construction
for *The Devil's Dictionary* (Ambrose Bierce, 1911; Project Gutenberg #972).

Source: https://www.gutenberg.org/ebooks/972
Raw text: https://www.gutenberg.org/cache/epub/972/pg972.txt
"""
from __future__ import annotations

import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import TensorDataset

DATA_URL = "https://www.gutenberg.org/cache/epub/972/pg972.txt"
SOURCE_PAGE = "https://www.gutenberg.org/ebooks/972"

PAD, UNK, NL, PAR = "<pad>", "<unk>", "<nl>", "<par>"
SPECIALS = [PAD, UNK, NL, PAR]

# words (with internal apostrophes / hyphens) or single punctuation marks
TOKEN_RE = re.compile(r"[a-z]+(?:['\-][a-z]+)*|--|[.,;:!?()\"]|<nl>|<par>")


# --------------------------------------------------------------------------- #
# 1. Loading & cleaning
# --------------------------------------------------------------------------- #
def download(path: str = "data/devils_dictionary.txt") -> str:
    """Download the raw Gutenberg text if it is not already on disk."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_raw(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def strip_gutenberg(raw: str) -> str:
    """Keep only the dictionary body: from the first entry (ABASEMENT) to the
    Gutenberg END marker. The licence header/footer and the author's preface
    are discarded because they are not part of the dictionary genre."""
    start = raw.find("ABASEMENT, n.")
    end = raw.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    return raw[start:end]


def clean_text(body: str) -> str:
    """Normalise the dictionary body into a single string in which
    paragraph breaks are marked with <par> and verse line breaks with <nl>."""
    body = body.replace("\r\n", "\n")
    # normalise typographic quotes / dashes
    body = (body.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2014", "--"))
    body = body.replace("_", "")                     # italics markers
    lines = body.split("\n")

    out_paragraphs, current, is_verse = [], [], False
    for line in lines:
        if re.fullmatch(r"\s*[A-Z]\s*", line):       # alphabet section headers "B", "C" ...
            continue
        if not line.strip():                          # blank line -> paragraph break
            if current:
                joiner = f" {NL} " if is_verse else " "
                out_paragraphs.append(joiner.join(current))
                current = []
            continue
        if not current:
            is_verse = line.startswith("  ")          # indented block = verse / quotation
        current.append(line.strip())
    if current:
        out_paragraphs.append((f" {NL} " if is_verse else " ").join(current))

    text = f" {PAR} ".join(out_paragraphs)
    return text.lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    """Human-readable rendering of a token list."""
    s = " ".join(tokens)
    s = s.replace(f" {PAR} ", "\n\n").replace(PAR, "\n\n")
    s = s.replace(f" {NL} ", "\n    ").replace(NL, "\n    ")
    s = re.sub(r" ([.,;:!?)])", r"\1", s)
    s = re.sub(r"\( ", "(", s)
    s = s.replace(" -- ", "--")
    return s


# --------------------------------------------------------------------------- #
# 2. Vocabulary
# --------------------------------------------------------------------------- #
@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids]

    @classmethod
    def build(cls, tokens: list[str], min_freq: int = 2) -> "Vocab":
        counts = Counter(tokens)
        words = sorted([w for w, c in counts.items() if c >= min_freq and w not in SPECIALS],
                       key=lambda w: (-counts[w], w))
        return cls(SPECIALS + words)


# --------------------------------------------------------------------------- #
# 3. Autoregressive input -> target sequences
# --------------------------------------------------------------------------- #
def make_windows(ids: np.ndarray, seq_len: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window: X = ids[i:i+seq_len], y = ids[i+seq_len], for i = 0, stride, 2*stride ..."""
    n = len(ids) - seq_len
    if n <= 0:
        return np.empty((0, seq_len), dtype=np.int64), np.empty((0,), dtype=np.int64)
    starts = np.arange(0, n, stride)
    idx = np.arange(seq_len)[None, :] + starts[:, None]
    return ids[idx], ids[starts + seq_len]


def block_split(ids: np.ndarray, seq_len: int, block_size: int = 2000,
                unk_id: int | None = None, train_stride: int = 1):
    """Split the token stream into contiguous blocks; block k goes to
    train (k%10 in 0..7), validation (k%10==8) or test (k%10==9).
    Windows are built *inside* blocks only, so no window straddles two
    splits (prevents leakage) and every split covers the whole alphabet.
    Windows whose target is <unk> are dropped (predicting <unk> is meaningless).
    Training windows use `train_stride` (CPU budget); val/test always use stride 1."""
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for k, start in enumerate(range(0, len(ids), block_size)):
        block = ids[start:start + block_size]
        name = "val" if k % 10 == 8 else "test" if k % 10 == 9 else "train"
        X, y = make_windows(block, seq_len, train_stride if name == "train" else 1)
        if unk_id is not None:
            keep = y != unk_id
            X, y = X[keep], y[keep]
        splits[name][0].append(X)
        splits[name][1].append(y)
    out = {}
    for name, (xs, ys) in splits.items():
        X = torch.from_numpy(np.concatenate(xs)).long()
        y = torch.from_numpy(np.concatenate(ys)).long()
        out[name] = TensorDataset(X, y)
    return out


@dataclass
class Corpus:
    text: str
    tokens: list[str]
    vocab: Vocab
    ids: np.ndarray
    datasets: dict


def prepare_corpus(path: str = "data/devils_dictionary.txt", seq_len: int = 20,
                   min_freq: int = 2, block_size: int = 2000, train_stride: int = 2) -> Corpus:
    raw = load_raw(download(path))
    text = clean_text(strip_gutenberg(raw))
    tokens = tokenize(text)
    vocab = Vocab.build(tokens, min_freq=min_freq)
    ids = np.array(vocab.encode(tokens), dtype=np.int64)
    datasets = block_split(ids, seq_len, block_size, unk_id=vocab.stoi[UNK],
                           train_stride=train_stride)
    return Corpus(text, tokens, vocab, ids, datasets)


def corpus_stats(c: Corpus) -> dict:
    counts = Counter(c.tokens)
    entries = re.findall(r"(?:^|<par> )([a-z][a-z' \-]*), (n|v|adj|adv|pp|p|prep|pron|conj|interj|x)\.",
                         c.text)
    words_only = [t for t in c.tokens if re.fullmatch(r"[a-z].*", t)]
    unk_rate = float(np.mean(c.ids == c.vocab.stoi[UNK]))
    return {
        "characters (cleaned)": len(c.text),
        "total tokens": len(c.tokens),
        "word tokens": len(words_only),
        "unique tokens (types)": len(counts),
        "vocab size (min_freq applied)": len(c.vocab),
        "hapax legomena (freq=1)": sum(1 for v in counts.values() if v == 1),
        "<unk> rate after vocab cap": round(unk_rate, 4),
        "dictionary entries (approx.)": len(entries),
        "verse line breaks (<nl>)": counts[NL],
        "paragraphs (<par>)": counts[PAR] + 1,
        "train sequences": len(c.datasets["train"]),
        "val sequences": len(c.datasets["val"]),
        "test sequences": len(c.datasets["test"]),
    }
