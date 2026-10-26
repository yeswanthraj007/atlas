"""Deep models that read raw price history (pre-registered:
research/preregistered/2026-10-26-deep-patterns-v1.md).

* ChartCNN: Jiang, Kelly & Xiu (2023, J. Finance) style. Each sample is a 64 x 60 black-
  and-white image of the last 20 trading days: per day a 3-pixel-wide OHLC bar (open tick,
  high-low line, close tick), a 20-day moving-average line, and volume bars at the bottom.
* PatchTransformer: PatchTST style (Nie et al. 2023). The last 120 days of log returns
  and log volume ratios, cut into 10-day patches (tokens), through a transformer encoder.

Both predict whether a stock's next 20-day return beats the cross-sectional median.
Every input uses data up to and including the sample date only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from atlas.ml.text_embed import device

IMG_DAYS, IMG_H, IMG_W = 20, 64, 60
PRICE_ROWS, VOL_ROWS = 51, 12  # 51 price rows, 1 gap row, 12 volume rows
SEQ_DAYS, PATCH = 120, 10
HORIZON = 20


def chart_image(
    o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, v: np.ndarray, ma: np.ndarray
) -> np.ndarray:
    """One 64 x 60 uint8 image (255 = ink) from 20 days of OHLC, volume and moving average.
    Row 0 is the TOP of the chart, so higher prices get smaller row numbers."""
    img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    lo_all = np.nanmin(np.r_[lo, ma])
    hi_all = np.nanmax(np.r_[h, ma])
    span = hi_all - lo_all if hi_all > lo_all else 1.0

    def row(p: float) -> int:
        return int(round((PRICE_ROWS - 1) * (1 - (p - lo_all) / span)))  # noqa: RUF046 (numpy)

    vmax = np.nanmax(v) if np.nanmax(v) > 0 else 1.0
    for i in range(IMG_DAYS):
        x = 3 * i
        img[row(o[i]), x] = 255
        img[row(h[i]) : row(lo[i]) + 1, x + 1] = 255
        img[row(c[i]), x + 2] = 255
        if np.isfinite(ma[i]):
            img[row(ma[i]), x + 1] = 255
        bar = int(round(VOL_ROWS * v[i] / vmax))  # noqa: RUF046 (numpy float)
        if bar > 0:
            img[IMG_H - bar :, x + 1] = 255
    return img


@dataclass
class Samples:
    keys: pd.DataFrame  # date, instrument
    images: np.ndarray  # packed bits, n x (64*60/8)
    sequences: np.ndarray  # n x 120 x 2 float16
    target: np.ndarray  # 1 if next-20d return beats the cross-sectional median, NaN unknown
    fwd_return: np.ndarray  # next-20d return (for evaluation only)
    label_end: pd.DatetimeIndex  # date the label becomes known


def build_samples(
    bars: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, stride: int = 5
) -> Samples:
    """Samples every `stride` trading days for every stock with enough history."""
    keys, imgs, seqs, fwd = [], [], [], []
    sample_days = dates[SEQ_DAYS::stride]
    pos_of = {d: i for i, d in enumerate(dates)}
    for inst, df in bars.items():
        df = df.reindex(dates)
        o, h, lo, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        ma = pd.Series(c).rolling(20, min_periods=20).mean().to_numpy()
        logret = np.log(c[1:] / c[:-1])
        logret = np.r_[np.nan, logret]
        vol_ratio = np.log(v / pd.Series(v).rolling(20, min_periods=10).mean().to_numpy())
        for d in sample_days:
            t = pos_of[d]
            win = slice(t - IMG_DAYS + 1, t + 1)
            seq_win = slice(t - SEQ_DAYS + 1, t + 1)
            if np.isnan(c[seq_win]).any() or np.isnan(o[win]).any() or np.isnan(ma[win]).any():
                continue
            img = chart_image(o[win], h[win], lo[win], c[win], v[win], ma[win])
            s = np.stack([logret[seq_win], vol_ratio[seq_win]], axis=1)
            if not np.isfinite(s).all():
                continue
            s = (s - s.mean(axis=0)) / (s.std(axis=0) + 1e-8)
            r = c[t + HORIZON] / c[t] - 1.0 if t + HORIZON < len(c) else np.nan
            keys.append((d, inst))
            imgs.append(np.packbits(img > 0))
            seqs.append(s.astype(np.float16))
            fwd.append(r)
    keys_df = pd.DataFrame(keys, columns=["date", "instrument"])
    fwd_a = np.array(fwd, dtype=float)
    med = pd.Series(fwd_a).groupby(keys_df["date"].to_numpy()).transform("median").to_numpy()
    target = np.where(np.isfinite(fwd_a), (fwd_a > med).astype(float), np.nan)
    pos = keys_df["date"].map(pos_of).to_numpy() + HORIZON
    valid = pos < len(dates)
    label_end = pd.Series(pd.NaT, index=range(len(pos)), dtype=dates.dtype)
    label_end[valid] = dates[pos[valid]]
    label_end = pd.DatetimeIndex(label_end)
    return Samples(keys_df, np.stack(imgs), np.stack(seqs), target, fwd_a, label_end)


def unpack_images(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=1)[:, : IMG_H * IMG_W].reshape(-1, 1, IMG_H, IMG_W)


class ChartCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, (5, 3), padding=(2, 1)),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d((2, 1)),
            )

        self.features = nn.Sequential(block(1, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * (IMG_H // 8) * IMG_W, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class PatchTransformer(nn.Module):
    def __init__(self, d_model: int = 64, heads: int = 4, layers: int = 3) -> None:
        super().__init__()
        n_tokens = SEQ_DAYS // PATCH
        self.embed = nn.Linear(PATCH * 2, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: batch x 120 x 2
        b = x.shape[0]
        tokens = x.reshape(b, SEQ_DAYS // PATCH, PATCH * 2)
        z = torch.cat([self.cls.expand(b, -1, -1), self.embed(tokens)], dim=1) + self.pos
        return self.head(self.encoder(z)[:, 0])


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch: int = 512
    epochs: int = 20
    patience: int = 3
    seed: int = 0
    history: list[tuple[int, float, float]] = field(default_factory=list)


def _batches(n: int, size: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, size):
        yield order[i : i + size]


def _inputs(kind: str, s: Samples, idx: np.ndarray) -> torch.Tensor:
    if kind == "cnn":
        return torch.tensor(unpack_images(s.images[idx]).astype(np.float32))
    return torch.tensor(s.sequences[idx].astype(np.float32))


def train(
    model: nn.Module,
    kind: str,
    s: Samples,
    tr: np.ndarray,
    va: np.ndarray,
    cfg: TrainConfig,
    progress=print,
) -> nn.Module:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    y = torch.tensor(np.nan_to_num(s.target), dtype=torch.long)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        for b in _batches(len(tr), cfg.batch, True, rng):
            ix = tr[b]
            opt.zero_grad()
            loss = loss_fn(model(_inputs(kind, s, ix).to(dev)), y[ix].to(dev))
            loss.backward()
            opt.step()
        model.eval()
        vloss, vacc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in _batches(len(va), 4096, False, rng):
                ix = va[b]
                out = model(_inputs(kind, s, ix).to(dev))
                yy = y[ix].to(dev)
                vloss += float(loss_fn(out, yy)) * len(ix)
                vacc += float((out.argmax(1) == yy).float().sum())
                n += len(ix)
        vloss, vacc = vloss / n, vacc / n
        cfg.history.append((epoch, vloss, vacc))
        progress(f"    epoch {epoch}: validation loss {vloss:.4f}, accuracy {vacc:.3%}")
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict(model: nn.Module, kind: str, s: Samples, idx: np.ndarray) -> np.ndarray:
    dev = device()
    out = []
    for i in range(0, len(idx), 4096):
        logits = model(_inputs(kind, s, idx[i : i + 4096]).to(dev))
        out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
