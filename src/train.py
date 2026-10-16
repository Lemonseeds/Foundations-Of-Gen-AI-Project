"""
Identical training / evaluation loop for all four models.
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import build_model, count_params


@dataclass
class TrainConfig:
    seq_len: int = 20
    train_stride: int = 2
    emb_dim: int = 128
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.3
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 0.0
    clip: float = 1.0
    max_epochs: int = 15
    patience: int = 3
    seed: int = 42
    threads: int = 6

    def model_hparams(self):
        return dict(emb_dim=self.emb_dim, hidden=self.hidden,
                    layers=self.layers, dropout=self.dropout)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device="cpu") -> dict:
    """Mean cross-entropy, perplexity = exp(CE), top-1 and top-5 accuracy."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    tot_loss, tot_n, top1, top5 = 0.0, 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        tot_loss += crit(logits, y).item()
        tot_n += y.numel()
        t5 = logits.topk(5, dim=-1).indices
        top1 += (t5[:, 0] == y).sum().item()
        top5 += (t5 == y[:, None]).any(dim=-1).sum().item()
    loss = tot_loss / tot_n
    return {"loss": loss, "perplexity": math.exp(loss),
            "top1_acc": top1 / tot_n, "top5_acc": top5 / tot_n}


def train_model(name: str, corpus, cfg: TrainConfig, device="cpu",
                save_dir: str | None = "models", verbose: bool = True):
    """Train one model with early stopping on validation loss.
    Returns (best_model, history list, summary dict)."""
    set_seed(cfg.seed)                                    # same init seed / data order for every model
    model = build_model(name, len(corpus.vocab), **cfg.model_hparams()).to(device)

    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(corpus.datasets["train"], batch_size=cfg.batch_size,
                          shuffle=True, generator=g)
    val_dl = DataLoader(corpus.datasets["val"], batch_size=512)
    test_dl = DataLoader(corpus.datasets["test"], batch_size=512)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    crit = nn.CrossEntropyLoss()

    history, best_val, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    total_train_time = 0.0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, run_n, run_correct = 0.0, 0, 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            run_loss += loss.item() * y.numel()
            run_n += y.numel()
            run_correct += (logits.argmax(-1) == y).sum().item()
        epoch_time = time.perf_counter() - t0
        total_train_time += epoch_time

        val = evaluate(model, val_dl, device)
        sched.step(val["loss"])
        tr_loss = run_loss / run_n
        rec = {"model": name, "epoch": epoch,
               "train_loss": tr_loss, "train_ppl": math.exp(tr_loss),
               "train_acc": run_correct / run_n,
               "val_loss": val["loss"], "val_ppl": val["perplexity"],
               "val_top1": val["top1_acc"], "val_top5": val["top5_acc"],
               "epoch_time_s": epoch_time, "lr": opt.param_groups[0]["lr"]}
        history.append(rec)
        if verbose:
            print(f"[{name:9s}] ep {epoch:2d} | train {tr_loss:.3f} (ppl {math.exp(tr_loss):7.1f}) "
                  f"| val {val['loss']:.3f} (ppl {val['perplexity']:7.1f}, top1 {val['top1_acc']:.3f}) "
                  f"| {epoch_time:5.1f}s")

        if val["loss"] < best_val - 1e-4:
            best_val, best_epoch, bad = val["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"[{name:9s}] early stop (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    test = evaluate(model, test_dl, device)
    val = evaluate(model, val_dl, device)
    summary = {"model": name, "params": count_params(model),
               "epochs_run": len(history), "best_epoch": best_epoch,
               "val_loss": val["loss"], "val_ppl": val["perplexity"],
               "test_loss": test["loss"], "test_ppl": test["perplexity"],
               "test_top1_acc": test["top1_acc"], "test_top5_acc": test["top5_acc"],
               "total_train_time_s": total_train_time,
               "time_per_epoch_s": total_train_time / len(history)}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "name": name,
                    "config": asdict(cfg), "vocab": corpus.vocab.itos},
                   os.path.join(save_dir, f"{name}.pt"))
    return model, history, summary
