"""
Four recurrent next-word language models sharing one interface:

    Embedding -> recurrent core (2 layers) -> dropout -> Linear(vocab)

The only thing that changes between models is the recurrent core, so the
comparison is controlled.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentLM(nn.Module):
    CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, vocab_size: int, cell: str = "lstm", emb_dim: int = 128,
                 hidden: int = 256, layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False, pad_idx: int = 0):
        super().__init__()
        self.cell_name = cell
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        rnn_cls = self.CELLS[cell]
        kwargs = dict(input_size=emb_dim, hidden_size=hidden, num_layers=layers,
                      dropout=dropout if layers > 1 else 0.0, batch_first=True,
                      bidirectional=bidirectional)
        if cell == "rnn":
            kwargs["nonlinearity"] = "tanh"
        self.rnn = rnn_cls(**kwargs)
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) token ids of the context window -> (B, V) next-token logits.

        Unidirectional: the last time step's hidden state summarises x[0..T-1].
        Bidirectional: we concatenate the forward state at the last step with
        the backward state at the first step (the backward pass has read the
        whole window right-to-left). Both summaries use ONLY the context
        window, so the next word (the target) is never visible -> causal.
        """
        emb = self.drop(self.embedding(x))
        out, _ = self.rnn(emb)                       # (B, T, H*dirs)
        if self.bidirectional:
            H = out.size(-1) // 2
            feat = torch.cat([out[:, -1, :H], out[:, 0, H:]], dim=-1)
        else:
            feat = out[:, -1, :]
        return self.fc(self.drop(feat))


MODEL_SPECS = {
    "SimpleRNN": dict(cell="rnn", bidirectional=False),
    "BiRNN":     dict(cell="rnn", bidirectional=True),
    "LSTM":      dict(cell="lstm", bidirectional=False),
    "GRU":       dict(cell="gru", bidirectional=False),
}


def build_model(name: str, vocab_size: int, **hparams) -> RecurrentLM:
    return RecurrentLM(vocab_size, **MODEL_SPECS[name], **hparams)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
