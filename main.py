"""
Command-line entry point: runs the whole assignment pipeline
(data -> 4 models -> evaluation -> temperature study -> plots) and writes
everything to results/ and models/.

    python main.py                 # full run
    python main.py --epochs 2      # quick smoke run
"""
import argparse
import os

import torch

from src.data import prepare_corpus
from src.experiment import (describe_dataset, plot_comparison, plot_dataset,
                            plot_temperature, plot_training, temperature_study,
                            train_all)
from src.train import TrainConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=TrainConfig.max_epochs)
    p.add_argument("--threads", type=int, default=TrainConfig.threads)
    args = p.parse_args()

    cfg = TrainConfig(max_epochs=args.epochs, threads=args.threads)
    torch.set_num_threads(cfg.threads)

    corpus = prepare_corpus(seq_len=cfg.seq_len, train_stride=cfg.train_stride)
    print(describe_dataset(corpus).to_string(index=False))
    plot_dataset(corpus)

    models, hist, summ = train_all(corpus, cfg)
    plot_training(hist)
    plot_comparison(summ)
    print("\n", summ.round(4).to_string())

    tdf, _ = temperature_study(models, corpus, seq_len=cfg.seq_len)
    plot_temperature(tdf)
    print("\n", tdf.round(3).to_string(index=False))
    print(f"\nAll outputs written to {os.path.abspath('results')}")


if __name__ == "__main__":
    main()
