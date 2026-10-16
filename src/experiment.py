"""
End-to-end experiment helpers shared by `main.py` and the notebook:
training all four models, plotting, and the temperature study.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch

from .data import NL, PAR, UNK, corpus_stats, tokenize
from .generate import corpus_ngram_sets, generate, render, text_metrics
from .models import MODEL_SPECS
from .train import TrainConfig, train_model

RESULTS = "results"
COLORS = {"SimpleRNN": "#e76f51", "BiRNN": "#f4a261", "LSTM": "#2a9d8f", "GRU": "#264653"}
SEEDS = ["politics, n.", "love, n.", "a lawyer is"]
TEMPERATURES = [0.2, 0.5, 1.0, 1.5]


def _savefig(fig, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
# Dataset description
# --------------------------------------------------------------------------- #
def describe_dataset(corpus) -> pd.DataFrame:
    stats = corpus_stats(corpus)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return pd.DataFrame(stats.items(), columns=["statistic", "value"])


def plot_dataset(corpus):
    counts = Counter(t for t in corpus.tokens if re.fullmatch(r"[a-z].*", t))
    top = counts.most_common(25)
    pos = Counter(re.findall(r"(?:^|<par> )[a-z][a-z' \-]*, ([a-z]+(?:\.[a-z]+)?)\.", corpus.text))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].barh([w for w, _ in top][::-1], [c for _, c in top][::-1], color="#2a9d8f")
    ax[0].set_title("25 most frequent words")
    freqs = sorted(counts.values(), reverse=True)
    ax[1].loglog(range(1, len(freqs) + 1), freqs, color="#264653")
    ax[1].set_title("Zipf plot (rank vs frequency)"); ax[1].set_xlabel("rank"); ax[1].set_ylabel("freq")
    common = pos.most_common(8)
    ax[2].bar([p for p, _ in common], [c for _, c in common], color="#e76f51")
    ax[2].set_title("Part-of-speech tags of headwords")
    fig.suptitle("The Devil's Dictionary - corpus characteristics", fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "dataset_overview.png")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_all(corpus, cfg: TrainConfig, names=tuple(MODEL_SPECS)):
    models, histories, summaries = {}, [], []
    for name in names:
        m, h, s = train_model(name, corpus, cfg)
        models[name] = m
        histories += h
        summaries.append(s)
    hist = pd.DataFrame(histories)
    summ = pd.DataFrame(summaries).set_index("model")
    os.makedirs(RESULTS, exist_ok=True)
    hist.to_csv(os.path.join(RESULTS, "training_history.csv"), index=False)
    summ.to_csv(os.path.join(RESULTS, "model_metrics.csv"))
    with open(os.path.join(RESULTS, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return models, hist, summ


def plot_training(hist: pd.DataFrame):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    for name, g in hist.groupby("model", sort=False):
        c = COLORS[name]
        ax[0].plot(g.epoch, g.train_loss, "--", color=c, alpha=.6)
        ax[0].plot(g.epoch, g.val_loss, "-o", color=c, ms=3, label=name)
        ax[1].plot(g.epoch, g.val_ppl, "-o", color=c, ms=3, label=name)
        ax[2].plot(g.epoch, g.val_top1, "-o", color=c, ms=3, label=name)
        ax[3].plot(g.epoch, g.epoch_time_s, "-o", color=c, ms=3, label=name)
    ax[0].set_title("Loss (solid = val, dashed = train)")
    ax[1].set_title("Validation perplexity")
    ax[2].set_title("Validation top-1 accuracy")
    ax[3].set_title("Seconds per epoch")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, "training_curves.png")


def plot_comparison(summ: pd.DataFrame):
    names = list(summ.index)
    cols = [COLORS[n] for n in names]
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, (col, title) in zip(ax, [("test_ppl", "Test perplexity (lower better)"),
                                   ("test_top1_acc", "Test top-1 accuracy"),
                                   ("test_top5_acc", "Test top-5 accuracy"),
                                   ("total_train_time_s", "Total training time (s)")]):
        bars = a.bar(names, summ[col], color=cols)
        a.bar_label(bars, fmt="%.3f" if "acc" in col else "%.1f", fontsize=8)
        a.set_title(title); a.grid(axis="y", alpha=.3)
    fig.tight_layout()
    return _savefig(fig, "model_comparison.png")


# --------------------------------------------------------------------------- #
# Temperature study
# --------------------------------------------------------------------------- #
def temperature_study(models: dict, corpus, seeds=SEEDS, temps=TEMPERATURES,
                      n_samples: int = 5, n_tokens: int = 60, seq_len: int = 20,
                      rng_seed: int = 0):
    """For every model x temperature: n_samples continuations of each seed.
    Returns metrics DataFrame and a nested dict of rendered example texts."""
    for s in seeds:
        oov = [t for t in tokenize(s.lower()) if t not in corpus.vocab.stoi]
        assert not oov, f"seed '{s}' contains out-of-vocabulary words: {oov}"
    bigrams, fourgrams = corpus_ngram_sets(corpus.tokens)
    rows, examples = [], {}
    for name, model in models.items():
        for T in temps:
            g = torch.Generator().manual_seed(rng_seed)       # identical RNG stream for every model
            all_samples = []
            for seed in seeds:
                outs = [generate(model, corpus.vocab, seed, n_tokens, T, seq_len, g)
                        for _ in range(n_samples)]
                all_samples += outs
                examples.setdefault(name, {}).setdefault(T, {})[seed] = render(seed, outs[0])
            rows.append({"model": name, "temperature": T,
                         **text_metrics(all_samples, bigrams, fourgrams)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "temperature_metrics.csv"), index=False)
    write_samples_md(examples, seeds, temps)
    return df, examples


def write_samples_md(examples, seeds, temps, path=os.path.join(RESULTS, "generated_samples.md")):
    lines = ["# Generated text samples\n",
             "First of 5 samples per (model, temperature, seed); 60 generated tokens each.\n",
             "Seed text is shown in **bold**.\n"]
    for seed in seeds:
        lines.append(f"\n## Seed: `{seed}`\n")
        for T in temps:
            lines.append(f"\n### Temperature {T}\n")
            for name in examples:
                txt = examples[name][T][seed]
                n = len(seed)
                body = txt[n:].strip().replace("\n", "<br>")
                lines.append(f"- **{name}**: **{txt[:n]}** {body}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return path


def plot_temperature(df: pd.DataFrame):
    metrics = [("distinct_2", "Diversity: distinct-2 (higher = more varied)"),
               ("rep_3gram", "Repetition: repeated 3-gram rate (lower better)"),
               ("bigram_attested", "Coherence proxy: bigrams seen in corpus"),
               ("novel_4gram", "Novelty: 4-grams not in corpus")]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    for a, (m, title) in zip(ax, metrics):
        for name, g in df.groupby("model", sort=False):
            a.plot(g.temperature, g[m], "-o", color=COLORS[name], label=name)
        a.set_title(title, fontsize=10); a.set_xlabel("temperature"); a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout()
    return _savefig(fig, "temperature_analysis.png")
