"""A language model that reads earnings press releases (pre-registered:
research/preregistered/2026-11-07-llm-reader-v1.md).

RoBERTa-base (training text ends in 2019, so it cannot know the 2024-2026 test window) is
fine-tuned end to end with three heads: reaction sign (T1), post-reaction drift rank (T2),
and move size versus the quarter median (T3). The model only ever sees text, with the
company's own name and ticker masked.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from atlas.ml.text_embed import device

MODEL = "FacebookAI/roberta-base"
REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
MAX_LEN = 384  # fixed shape: on the Apple GPU every new shape triggers a slow recompile
MAX_CHARS = 2600  # the first ~2,600 characters comfortably fill 384 tokens


# --------------------------------------------------------------------------------------
# Text preparation
# --------------------------------------------------------------------------------------


def strip_header(text: str) -> str:
    """Drop the exhibit header lines (EX-99.1, sequence number, file name) before the body."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and len(lines[i].strip()) < 30:
        i += 1
    return "\n".join(lines[i:])


def mask_company(text: str, names: list[str], ticker: str) -> str:
    """Replace the company's own names and ticker with [COMPANY]."""
    parts = [re.escape(n) for n in sorted(set(names), key=len, reverse=True) if n]
    out = text
    if parts:
        out = re.sub("|".join(parts), "[COMPANY]", out, flags=re.I)
    if ticker:
        out = re.sub(rf"(?<![A-Za-z]){re.escape(ticker)}(?![A-Za-z])", "[COMPANY]", out)
    return out


def prepare_text(raw: str, names: list[str], ticker: str, max_chars: int = MAX_CHARS) -> str:
    body = re.sub(r"[ \t]+", " ", strip_header(raw))
    body = re.sub(r"\n{2,}", "\n", body)
    return mask_company(body[: max_chars + 400], names, ticker)[:max_chars]


def tokenize(texts: list[str], tok) -> tuple[np.ndarray, np.ndarray]:
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="np")
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int8)


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------


def drift_labels(
    closes: pd.DataFrame, benchmark: pd.Series, d0: np.ndarray, stock: np.ndarray, h: int = 20
) -> np.ndarray:
    """Market-adjusted return from the close of the reaction session (d0) to h days later.
    NaN when the outcome is not yet known. Uses only prices AFTER the decision point."""
    c = closes.to_numpy()
    b = benchmark.reindex(closes.index).to_numpy()
    ok = d0 + h < len(closes)
    out = np.full(len(d0), np.nan)
    i = np.where(ok)[0]
    out[i] = (c[d0[i] + h, stock[i]] / c[d0[i], stock[i]] - 1.0) - (b[d0[i] + h] / b[d0[i]] - 1.0)
    return out


def gaussian_rank_by_group(values: pd.Series, group: pd.Series) -> np.ndarray:
    from scipy.stats import norm

    pct = values.groupby(group.to_numpy()).rank(pct=True)
    n = values.groupby(group.to_numpy()).transform("count")
    return np.asarray(norm.ppf((pct * n - 0.5) / n), dtype=np.float32)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class ReaderModel(nn.Module):
    """Encoder plus three heads on the first token: [T1 logit, T2 score, T3 logit]."""

    def __init__(self, encoder: nn.Module, width: int = 768) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(width, 3))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(h)


def load_reader() -> ReaderModel:
    from transformers import AutoModel

    enc = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    return ReaderModel(enc, enc.config.hidden_size)


def reader_loss(out: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, y3: torch.Tensor):
    """BCE(T1) + MSE(T2) + BCE(T3), equal weights (fixed in the pre-registration)."""
    bce = nn.functional.binary_cross_entropy_with_logits
    return bce(out[:, 0], y1) + nn.functional.mse_loss(out[:, 1], y2) + bce(out[:, 2], y3)


@dataclass
class ReaderConfig:
    lr: float = 2e-5
    batch: int = 16
    epochs: int = 3
    warmup: float = 0.06
    weight_decay: float = 0.01
    seed: int = 0
    history: list[tuple[int, float, float]] = field(default_factory=list)


def _batch(split: dict, ix: np.ndarray, dev):
    return (
        torch.from_numpy(split["ids"][ix].astype(np.int64)).to(dev),
        torch.from_numpy(split["mask"][ix].astype(np.int64)).to(dev),
        *(torch.from_numpy(split[k][ix].astype(np.float32)).to(dev) for k in ("y1", "y2", "y3")),
    )


@torch.no_grad()
def predict_reader(model: ReaderModel, split: dict, batch: int = 64) -> np.ndarray:
    dev = device()
    model.eval().to(dev)
    out = []
    for i in range(0, len(split["ids"]), batch):
        ix = np.arange(i, min(i + batch, len(split["ids"])))
        ids = torch.from_numpy(split["ids"][ix].astype(np.int64)).to(dev)
        mask = torch.from_numpy(split["mask"][ix].astype(np.int64)).to(dev)
        out.append(model(ids, mask).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def val_loss(model: ReaderModel, split: dict, batch: int = 64) -> float:
    dev = device()
    p = torch.from_numpy(predict_reader(model, split, batch)).to(dev)
    y = [torch.from_numpy(split[k].astype(np.float32)).to(dev) for k in ("y1", "y2", "y3")]
    return float(reader_loss(p, *y))


def train_reader(
    model: ReaderModel,
    train: dict,
    val: dict,
    cfg: ReaderConfig,
    ckpt_dir: str | Path,
    progress=print,
) -> ReaderModel:
    """Fine-tune; after every epoch validate and checkpoint (resumable). Keeps the epoch with
    the lowest validation loss."""
    ckpt = Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dev = device()
    model.to(dev)
    steps_per_epoch = len(train["ids"]) // cfg.batch
    total = steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
    )
    start, best = 0, math.inf
    state = ckpt / "state.pt"
    if state.exists():  # resume after an interruption, at epoch granularity
        st = torch.load(state, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, cfg.history = st["epoch"] + 1, st["best"], st["history"]
        progress(f"  resumed after epoch {st['epoch']}")
    for epoch in range(start, cfg.epochs):
        model.train()
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(train["ids"]))
        tot, n = 0.0, 0
        for s in range(steps_per_epoch):
            ids, mask, y1, y2, y3 = _batch(train, order[s * cfg.batch : (s + 1) * cfg.batch], dev)
            loss = reader_loss(model(ids, mask), y1, y2, y3)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n += 1
            if (s + 1) % 200 == 0:
                progress(
                    f"    epoch {epoch} step {s + 1}/{steps_per_epoch}: train loss {tot / n:.4f}"
                )
        vl = val_loss(model, val)
        cfg.history.append((epoch, tot / max(n, 1), vl))
        progress(f"  epoch {epoch}: train loss {tot / max(n, 1):.4f}, validation loss {vl:.4f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), ckpt / "best.pt")
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
             "epoch": epoch, "best": best, "history": cfg.history},
            state,
        )  # fmt: skip
    model.load_state_dict(torch.load(ckpt / "best.pt", map_location="cpu"))
    return model.to(dev).eval()
