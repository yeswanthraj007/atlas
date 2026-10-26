"""MarketGPT: a small decoder-only transformer over each stock's daily price, volume and
news (pre-registered: research/preregistered/2026-10-26-market-transformer-v2.md).

Same recipe as a language model, at our scale:
  * one token per stock-day; a causal mask, so day t only attends to days <= t;
  * stage 1, self-supervised pretraining: at every day predict the next day's return and
    absolute return (the analogue of next-token prediction);
  * stage 2, fine-tuning with a new head: does the next 20-day return beat the
    cross-sectional median? (or, for the volatility variant, next-20-day log volatility).

No look-ahead: every input feature on day t uses data up to day t's close only. Returns
are scaled by the trailing 60-day volatility (not by statistics of the whole window, which
would leak later days into earlier tokens). News published after day t's 16:00 ET close
belongs to day t+1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from atlas.ml.text_embed import device

CONTEXT = 120
HORIZON = 20
PRICE_FEATURES = [
    "ret_z",  # daily return / trailing 60-day vol
    "abs_ret_z",
    "gap_z",  # open vs previous close
    "range_z",  # log(high / low)
    "log_volume_ratio",  # volume vs trailing 20-day mean
    "log_vol_20_60",  # 20-day vs 60-day volatility
    "log_vol_level",  # log annualised 60-day volatility, centred
    "market_ret_z",  # SPY return / SPY trailing vol
]
NEWS_FEATURES = ["log_news_count", "no_news"]


@dataclass
class StockPanel:
    """Per-stock daily arrays aligned to one trading calendar."""

    instruments: list[str]
    dates: pd.DatetimeIndex
    price: np.ndarray  # stocks x days x len(PRICE_FEATURES), float32, NaN when unknown
    news_meta: np.ndarray  # stocks x days x 2
    news_emb: np.ndarray  # stocks x days x k (reduced text embedding; 0 when no news)
    next_ret_z: np.ndarray  # stocks x days: day t+1 return / vol60(t) (pretraining target)
    fwd_ret: np.ndarray  # stocks x days: return t -> t+20
    fwd_log_vol: np.ndarray  # stocks x days: log annualised vol over t+1..t+20


def price_features(bars: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Causal daily features for one stock (`bars` reindexed to the calendar)."""
    c, o, h, lo, v = (bars[k] for k in ("close", "open", "high", "low", "volume"))
    r = c.pct_change(fill_method=None)
    vol60 = r.rolling(60, min_periods=40).std()
    vol20 = r.rolling(20, min_periods=15).std()
    m = market.pct_change(fill_method=None)
    mvol = m.rolling(60, min_periods=40).std()
    f = pd.DataFrame(
        {
            "ret_z": r / vol60,
            "abs_ret_z": r.abs() / vol60,
            "gap_z": np.log(o / c.shift(1)) / vol60,
            "range_z": np.log(h / lo) / vol60,
            "log_volume_ratio": np.log(v / v.rolling(20, min_periods=10).mean()),
            "log_vol_20_60": np.log(vol20 / vol60),
            "log_vol_level": np.log(vol60 * np.sqrt(252)) + 1.2,
            "market_ret_z": m / mvol,
        }
    )
    return f.replace([np.inf, -np.inf], np.nan).clip(-10, 10)


def build_panel(
    bars: dict[str, pd.DataFrame],
    market: pd.Series,
    dates: pd.DatetimeIndex,
    news_daily: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    emb_dim: int = 0,
) -> StockPanel:
    """`news_daily[instrument] = (day positions, counts, mean embeddings)`."""
    insts = list(bars)
    n, t = len(insts), len(dates)
    price = np.full((n, t, len(PRICE_FEATURES)), np.nan, np.float32)
    meta = np.zeros((n, t, 2), np.float32)
    meta[:, :, 1] = 1.0
    emb = np.zeros((n, t, emb_dim), np.float16)
    nxt = np.full((n, t), np.nan, np.float32)
    fwd = np.full((n, t), np.nan, np.float32)
    fvol = np.full((n, t), np.nan, np.float32)
    mkt = market.reindex(dates)
    for i, inst in enumerate(insts):
        b = bars[inst].reindex(dates)
        f = price_features(b, mkt)
        price[i] = f[PRICE_FEATURES].to_numpy(np.float32)
        c = b["close"]
        r = c.pct_change(fill_method=None)
        vol60 = r.rolling(60, min_periods=40).std()
        nxt[i] = (r.shift(-1) / vol60).clip(-10, 10).to_numpy(np.float32)
        fwd[i] = (c.shift(-HORIZON) / c - 1.0).to_numpy(np.float32)
        fv = r[::-1].rolling(HORIZON, min_periods=int(HORIZON * 0.8)).std()[::-1].shift(-1)
        fvol[i] = np.log((fv * np.sqrt(252)).where(fv > 0)).to_numpy(np.float32)
        if news_daily and inst in news_daily:
            pos, cnt, e = news_daily[inst]
            meta[i, pos, 0] = np.log1p(cnt)
            meta[i, pos, 1] = 0.0
            emb[i, pos] = e
    return StockPanel(insts, dates, price, meta, emb, nxt, fwd, fvol)


def sample_index(
    panel: StockPanel, stride: int = 5, membership: pd.DataFrame | None = None
) -> pd.DataFrame:
    """(stock, end day) for every window of CONTEXT complete days, every `stride` days."""
    ok = np.isfinite(panel.price).all(axis=2)
    # complete[i, t] = the CONTEXT days ending at t are all complete
    run = np.zeros(ok.shape[1] + 1, np.int64)
    rows = []
    ends = np.arange(CONTEXT - 1, len(panel.dates), stride)
    mem = None
    if membership is not None:
        mem = (
            membership.reindex(index=panel.dates, columns=panel.instruments)
            .fillna(False)
            .to_numpy(bool)
        )
    for i in range(len(panel.instruments)):
        run[1:] = np.cumsum(ok[i])
        good = ends[(run[ends + 1] - run[ends + 1 - CONTEXT]) == CONTEXT]
        if mem is not None:
            good = good[mem[good, i]]
        rows.append(np.stack([np.full(len(good), i), good], axis=1))
    idx = np.concatenate(rows) if rows else np.zeros((0, 2), np.int64)
    df = pd.DataFrame(idx, columns=["stock", "end"])
    df["date"] = panel.dates[df["end"].to_numpy()]
    df["instrument"] = [panel.instruments[i] for i in df["stock"]]
    fwd = panel.fwd_ret[df["stock"], df["end"]]
    df["fwd_ret"] = fwd
    med = df.groupby("date")["fwd_ret"].transform("median")
    df["target"] = np.where(np.isfinite(fwd), (fwd > med).astype(float), np.nan)
    df["fwd_log_vol"] = panel.fwd_log_vol[df["stock"], df["end"]]
    lab = df["end"].to_numpy() + HORIZON
    df["label_end"] = pd.DatetimeIndex(
        [panel.dates[p] if p < len(panel.dates) else pd.NaT for p in lab]
    )
    return df


def gather(panel: StockPanel, stocks: np.ndarray, ends: np.ndarray, use_news: bool):
    """Model inputs for a batch: (price+meta, embeddings, next-day targets)."""
    days = ends[:, None] + np.arange(-CONTEXT + 1, 1)[None, :]
    s = stocks[:, None]
    x = np.concatenate([panel.price[s, days], panel.news_meta[s, days]], axis=2)
    e = panel.news_emb[s, days].astype(np.float32)
    if not use_news:
        x[:, :, len(PRICE_FEATURES) :] = 0.0
        e = np.zeros_like(e)
    y_next = panel.next_ret_z[s, days]
    return torch.from_numpy(x), torch.from_numpy(e), torch.from_numpy(y_next)


class MarketGPT(nn.Module):
    def __init__(self, emb_dim: int, width: int = 128, layers: int = 4, heads: int = 4) -> None:
        super().__init__()
        n_in = len(PRICE_FEATURES) + len(NEWS_FEATURES)
        self.inp = nn.Linear(n_in, width)
        self.news = nn.Linear(emb_dim, width, bias=False) if emb_dim else None
        self.pos = nn.Parameter(torch.randn(1, CONTEXT, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.next_day = nn.Linear(width, 2)  # pretraining: next return, next |return|
        self.head = nn.Linear(width, 2)  # fine-tuning (classification) ...
        self.vol_head = nn.Linear(width, 1)  # ... or volatility regression
        mask = torch.triu(torch.full((CONTEXT, CONTEXT), float("-inf")), diagonal=1)
        self.register_buffer("causal", mask, persistent=False)

    def hidden(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        z = self.inp(torch.nan_to_num(x))
        if self.news is not None:
            z = z + self.news(e)
        z = z + self.pos
        return self.norm(self.blocks(z, mask=self.causal, is_causal=True))


@dataclass
class GPTConfig:
    lr: float = 3e-4
    batch: int = 256
    pretrain_epochs: int = 10
    finetune_epochs: int = 20
    patience: int = 3
    warmup_steps: int = 500
    seed: int = 0
    history: list[tuple[str, int, float]] = field(default_factory=list)


def _schedule(opt: torch.optim.Optimizer, warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _loss(model: MarketGPT, stage: str, panel, rows: pd.DataFrame, use_news: bool, dev):
    x, e, y_next = gather(panel, rows["stock"].to_numpy(), rows["end"].to_numpy(), use_news)
    h = model.hidden(x.to(dev), e.to(dev))
    if stage == "pretrain":
        pred = model.next_day(h)
        y = y_next.to(dev)
        target = torch.stack([y, y.abs()], dim=-1)
        ok = torch.isfinite(target)
        return nn.functional.huber_loss(pred[ok], target[ok])
    last = h[:, -1]
    if stage == "classify":
        y = torch.tensor(rows["target"].to_numpy(), dtype=torch.long, device=dev)
        return nn.functional.cross_entropy(model.head(last), y)
    y = torch.tensor(rows["fwd_log_vol"].to_numpy(), dtype=torch.float32, device=dev)
    return nn.functional.mse_loss(model.vol_head(last).squeeze(-1), y)


def fit_stage(
    model: MarketGPT,
    stage: str,
    panel: StockPanel,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    cfg: GPTConfig,
    use_news: bool,
    progress=print,
) -> MarketGPT:
    """stage: "pretrain", "classify" or "vol". Early stopping on validation loss."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = device()
    model = model.to(dev)
    epochs = cfg.pretrain_epochs if stage == "pretrain" else cfg.finetune_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(train_rows) / cfg.batch)
    sched = _schedule(opt, cfg.warmup_steps, steps)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_rows))
        for i in range(0, len(order), cfg.batch):
            loss = _loss(
                model, stage, panel, train_rows.iloc[order[i : i + cfg.batch]], use_news, dev
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for i in range(0, len(val_rows), 1024):
                part = val_rows.iloc[i : i + 1024]
                tot += float(_loss(model, stage, panel, part, use_news, dev)) * len(part)
                n += len(part)
        vloss = tot / max(n, 1)
        cfg.history.append((stage, epoch, vloss))
        progress(f"    {stage} epoch {epoch}: validation loss {vloss:.4f}")
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
def predict_rows(
    model: MarketGPT, stage: str, panel: StockPanel, rows: pd.DataFrame, use_news: bool
) -> np.ndarray:
    """Up-probability ("classify") or next-20-day log volatility ("vol")."""
    dev = device()
    out = []
    for i in range(0, len(rows), 1024):
        part = rows.iloc[i : i + 1024]
        x, e, _ = gather(panel, part["stock"].to_numpy(), part["end"].to_numpy(), use_news)
        last = model.hidden(x.to(dev), e.to(dev))[:, -1]
        if stage == "classify":
            out.append(torch.softmax(model.head(last), dim=1)[:, 1].cpu().numpy())
        else:
            out.append(model.vol_head(last).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def qlike(log_vol_pred: np.ndarray, log_vol_true: np.ndarray) -> np.ndarray:
    """QLIKE loss on variances (Patton 2011): robust to noise in realised volatility."""
    ratio = np.exp(2 * (log_vol_true - log_vol_pred))
    return ratio - np.log(ratio) - 1.0
