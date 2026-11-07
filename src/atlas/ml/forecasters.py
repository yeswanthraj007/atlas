"""Forecasters for the shootout (pre-registered: research/preregistered/2026-11-07-forecaster-
shootout-v1.md): ARIMA-GARCH and LSTM, plus the scoring helpers.

ARMA(1,1) for the mean and GARCH(1,1) for the variance are fitted once per stock on
data to a cut-off date and then frozen. Forecasting at any later date is a closed-form
recursion over the full series (no refitting), so there is no look-ahead: the forecast at
position p uses only returns up to and including p.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.signal import lfilter
from torch import nn

from atlas.ml.market_gpt import gather
from atlas.ml.text_embed import device

H = 20
TRADING_DAYS = 252


# --------------------------------------------------------------------------------------
# ARMA(1,1) + GARCH(1,1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmaGarch:
    mu: float  # mean of the process (percent per day)
    phi: float
    theta: float
    omega: float  # variance (percent^2)
    alpha: float
    beta: float

    @property
    def long_run_var(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)


def fit_arma_garch(r_pct: np.ndarray) -> ArmaGarch | None:
    """Fit on daily returns in percent. None if the fit fails or is not stationary."""
    from arch import arch_model
    from statsmodels.tsa.arima.model import ARIMA

    r = np.asarray(r_pct, float)
    if len(r) < 500 or not np.isfinite(r).all():
        return None
    try:
        res = ARIMA(r, order=(1, 0, 1), trend="c").fit()
        mu, phi, theta = (float(res.params[i]) for i in (0, 1, 2))
        if abs(phi) >= 0.999 or abs(theta) >= 0.999:
            return None
        e = lfilter([1.0, -phi], [1.0, theta], r - mu)
        g = arch_model(e, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(
            disp="off", show_warning=False
        )
        omega, alpha, beta = (float(g.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
    except Exception:
        return None
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 0.9999):
        return None
    return ArmaGarch(mu, phi, theta, omega, alpha, beta)


def arma_garch_forecast(
    m: ArmaGarch, r_pct: np.ndarray, positions: np.ndarray, h: int = H
) -> tuple[np.ndarray, np.ndarray]:
    """At each position p (a day index into `r_pct`): the h-day return forecast (sum of
    daily log returns, as a fraction) and the h-day volatility forecast (annualised).
    Uses returns up to and including day p only."""
    r = np.asarray(r_pct, float)
    e = lfilter([1.0, -m.phi], [1.0, m.theta], r - m.mu)
    sbar = m.long_run_var
    x = np.empty(len(r) + 1)
    x[0] = m.omega + m.alpha * sbar
    x[1:] = m.omega + m.alpha * e**2
    v = lfilter([1.0], [1.0, -m.beta], x, zi=np.array([m.beta * sbar]))[0]  # v[t] = var of day t
    p = np.asarray(positions)
    v1 = v[p + 1]  # variance of day p+1, known at the close of p
    pers = m.alpha + m.beta
    k = np.arange(h)
    mean_var = sbar + (v1 - sbar) * (pers**k).sum() / h  # average daily variance over h days
    vol = np.sqrt(mean_var / 1e4 * TRADING_DAYS)
    m1 = m.mu + m.phi * (r[p] - m.mu) + m.theta * e[p]
    total_pct = h * m.mu + (m1 - m.mu) * (1.0 - m.phi**h) / (1.0 - m.phi)
    return total_pct / 100.0, vol


# --------------------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------------------


class LSTMBackbone(nn.Module):
    """Same interface as MarketGPT (`hidden(x, e)`, `norm`), so ScoreModel and fit_ranker
    train it unchanged."""

    def __init__(
        self, n_in: int = 10, width: int = 128, layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_in, width, layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(width)

    def hidden(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        z, _ = self.lstm(torch.nan_to_num(x))
        return self.norm(z)


class LSTMVol(nn.Module):
    def __init__(self, backbone: LSTMBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.norm.normalized_shape[0], 1)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.hidden(x, e)[:, -1]).squeeze(-1)


@dataclass
class VolConfig:
    lr: float = 1e-3
    batch: int = 512
    samples_per_epoch: int = 400_000
    epochs: int = 8
    patience: int = 2
    seed: int = 0
    history: list[tuple[int, float, float]] = field(default_factory=list)


def fit_vol(model: LSTMVol, panel, train_rows, val_rows, cfg: VolConfig, progress=print):
    """Regress log 20-day volatility; early stopping on validation MSE. Market input hidden."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = device()
    model = model.to(dev)
    tr = train_rows[train_rows["fwd_log_vol"].notna()].reset_index(drop=True)
    va = val_rows[val_rows["fwd_log_vol"].notna()]
    va = va[va["date"].isin(sorted(va["date"].unique())[::5])].reset_index(drop=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    best, best_state, bad = np.inf, None, 0

    def batch_loss(rows: pd.DataFrame) -> torch.Tensor:
        x, e, _ = gather(panel, rows["stock"].to_numpy(), rows["end"].to_numpy(), False, True)
        y = torch.tensor(rows["fwd_log_vol"].to_numpy(), dtype=torch.float32, device=dev)
        return nn.functional.mse_loss(model(x.to(dev), e.to(dev)), y)

    for epoch in range(cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for _ in range(max(1, cfg.samples_per_epoch // cfg.batch)):
            rows = tr.iloc[rng.integers(0, len(tr), cfg.batch)]
            loss = batch_loss(rows)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        model.eval()
        with torch.no_grad():
            vl = float(
                np.mean([float(batch_loss(va.iloc[i : i + 2048])) for i in range(0, len(va), 2048)])
            )
        cfg.history.append((epoch, tot / max(n, 1), vl))
        progress(f"    epoch {epoch}: train MSE {tot / max(n, 1):.4f}, validation MSE {vl:.4f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict_vol(model: LSTMVol, panel, rows: pd.DataFrame) -> np.ndarray:
    dev = device()
    model.eval()
    out = []
    for i in range(0, len(rows), 2048):
        part = rows.iloc[i : i + 2048]
        x, e, _ = gather(panel, part["stock"].to_numpy(), part["end"].to_numpy(), False, True)
        out.append(model(x.to(dev), e.to(dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def qlike(log_vol_pred: np.ndarray, log_vol_true: np.ndarray) -> np.ndarray:
    ratio = np.exp(2 * (log_vol_true - log_vol_pred))
    return ratio - np.log(ratio) - 1.0


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def score_return_task(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """df has date and fwd_ret; mean cross-sectional Spearman IC, its t, accuracy vs median."""
    d = df[["date", "fwd_ret"]].assign(p=pred)
    ic = d.groupby("date").apply(lambda g: stats.spearmanr(g["p"], g["fwd_ret"]).statistic)
    med = d.groupby("date")["fwd_ret"].transform("median")
    pm = d.groupby("date")["p"].transform("median")
    return {
        "ic": float(ic.mean()),
        "ic_t": t_stat(ic),
        "accuracy": float(((d["p"] > pm) == (d["fwd_ret"] > med)).mean()),
        "ic_by_date": ic,
    }


def score_vol_task(df: pd.DataFrame, pred_log_vol: np.ndarray, har_log_vol: np.ndarray) -> dict:
    """df has date and fwd_log_vol. QLIKE, Diebold-Mariano t against HAR, above-median accuracy."""
    d = df[["date", "fwd_log_vol"]].assign(p=pred_log_vol, h=har_log_vol)
    d["loss"] = qlike(d["p"].to_numpy(), d["fwd_log_vol"].to_numpy())
    d["loss_har"] = qlike(d["h"].to_numpy(), d["fwd_log_vol"].to_numpy())
    diff = d.groupby("date").apply(lambda g: (g["loss_har"] - g["loss"]).mean())
    acc = d.groupby("date").apply(
        lambda g: float(
            ((g["p"] > g["p"].median()) == (g["fwd_log_vol"] > g["fwd_log_vol"].median())).mean()
        )
    )
    return {
        "qlike": float(d["loss"].mean()),
        "dm_t_vs_har": t_stat(diff),  # > 0: better than HAR
        "accuracy": float(acc.mean()),
    }
