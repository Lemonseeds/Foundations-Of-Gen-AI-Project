"""
Autoregressive generation with temperature sampling, plus automatic
measures of diversity, repetition and (proxy) coherence of generated text.
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from .data import PAD, UNK, NL, PAR, tokenize, detokenize


@torch.no_grad()
def generate(model, vocab, seed_text: str, n_tokens: int = 60, temperature: float = 1.0,
             seq_len: int = 20, generator: torch.Generator | None = None,
             device="cpu") -> list[str]:
    """Generate n_tokens words one at a time, feeding each prediction back in.

    temperature -> 0 approaches greedy decoding; T=1 samples from the model's
    distribution; T>1 flattens it. <pad>/<unk> are never sampled.
    The model always sees at most the last `seq_len` tokens (its training
    context length); shorter seeds are fed as-is (no padding)."""
    model.eval()
    ids = vocab.encode(tokenize(seed_text.lower()))
    banned = torch.tensor([vocab.stoi[PAD], vocab.stoi[UNK]])
    generated = []
    for _ in range(n_tokens):
        x = torch.tensor([ids[-seq_len:]], device=device)
        logits = model(x)[0]
        logits[banned] = -float("inf")
        if temperature <= 1e-6:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        ids.append(nxt)
        generated.append(vocab.itos[nxt])
    return generated


# --------------------------------------------------------------------------- #
# Text-quality measures
# --------------------------------------------------------------------------- #
def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _words(tokens):
    """Drop structural tokens so metrics reflect the words themselves."""
    return [t for t in tokens if t not in (NL, PAR)]


def distinct_n(samples: list[list[str]], n: int) -> float:
    """Diversity: unique n-grams / total n-grams, pooled over samples."""
    grams = [g for s in samples for g in _ngrams(_words(s), n)]
    return len(set(grams)) / max(len(grams), 1)


def repetition_rate(samples: list[list[str]], n: int = 3) -> float:
    """Repetition: fraction of n-grams inside a sample that already occurred
    earlier in the same sample (averaged over samples). 0 = no loops."""
    rates = []
    for s in samples:
        grams = _ngrams(_words(s), n)
        if not grams:
            continue
        c = Counter(grams)
        rates.append(sum(v - 1 for v in c.values()) / len(grams))
    return sum(rates) / max(len(rates), 1)


def bigram_attestation(samples: list[list[str]], corpus_bigrams: set) -> float:
    """Coherence proxy: share of generated word bigrams that also occur in the
    training corpus. High = locally grammatical / plausible word order;
    very high + low distinct-n = copying; low = word salad."""
    grams = [g for s in samples for g in _ngrams(_words(s), 2)]
    return sum(g in corpus_bigrams for g in grams) / max(len(grams), 1)


def novel_4gram_rate(samples: list[list[str]], corpus_4grams: set) -> float:
    """Share of generated 4-grams NOT found in the corpus (memorisation check)."""
    grams = [g for s in samples for g in _ngrams(_words(s), 4)]
    return sum(g not in corpus_4grams for g in grams) / max(len(grams), 1)


def text_metrics(samples, corpus_bigrams, corpus_4grams) -> dict:
    return {"distinct_1": distinct_n(samples, 1),
            "distinct_2": distinct_n(samples, 2),
            "rep_3gram": repetition_rate(samples, 3),
            "bigram_attested": bigram_attestation(samples, corpus_bigrams),
            "novel_4gram": novel_4gram_rate(samples, corpus_4grams)}


def corpus_ngram_sets(tokens: list[str]):
    w = _words(tokens)
    return set(_ngrams(w, 2)), set(_ngrams(w, 4))


def render(seed: str, tokens: list[str]) -> str:
    return detokenize(tokenize(seed.lower()) + tokens)
