"""Volatility forecasting: how much will each stock move over the next month?

Unlike return direction, volatility is highly predictable (it clusters and persists).
The forecasts feed position sizing, VaR and risk limits.

Models
  * naive: next month's volatility = last month's (the benchmark to beat)
  * HAR:   log RV(next) ~ log RV(5d) + log RV(21d) + log RV(63d)   (Corsi 2009)
  * ridge / GBM on HAR terms + 1-year vol, market vol, size, recent move, earnings timing

Everything is in log annualised volatility; forecasts convert back with exp().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_realized_vol(closes: pd.DataFrame, window: int) -> pd.DataFrame:
    r = closes.pct_change(fill_method=None)
    rv = r.rolling(window, min_periods=max(3, int(window * 0.8))).std() * np.sqrt(TRADING_DAYS)
    return np.log(rv.where(rv > 0))


def forward_log_vol(closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Realised volatility of returns from t+1 .. t+horizon (fully in the future of t)."""
    r = closes.pct_change(fill_method=None)
    fwd = r[::-1].rolling(horizon, min_periods=int(horizon * 0.8)).std()[::-1].shift(-1)
    return np.log((fwd * np.sqrt(TRADING_DAYS)).where(fwd > 0))


def build_vol_dataset(
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
    horizon: int = 21,
    extra: dict[str, pd.DataFrame] | None = None,
    benchmark: pd.Series | None = None,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feats = {
        "lrv_5": log_realized_vol(closes, 5),
        "lrv_21": log_realized_vol(closes, 21),
        "lrv_63": log_realized_vol(closes, 63),
        "lrv_252": log_realized_vol(closes, 252),
        "abs_ret_21": (closes / closes.shift(21) - 1.0).abs(),
    }
    for k, v in (extra or {}).items():
        feats[k] = v.reindex_like(closes)
    target = forward_log_vol(closes, horizon)
    cols = {k: f.reindex(dates).stack(future_stack=True) for k, f in feats.items()}
    data = pd.DataFrame(cols)
    data["target"] = target.reindex(dates).stack(future_stack=True).reindex(data.index)
    data.index.names = ["date", "instrument"]
    data = data[data["lrv_21"].notna() & data["lrv_63"].notna()]
    if membership is not None:
        m = membership.reindex(index=dates, columns=closes.columns).fillna(False)
        keep = m.stack(future_stack=True).reindex(data.index).fillna(False).to_numpy(bool)
        data = data[keep]
    pos = closes.index.get_indexer(data.index.get_level_values("date")) + horizon
    data["label_end"] = [closes.index[p] if p < len(closes) else pd.NaT for p in pos]
    if benchmark is not None:
        b = benchmark.reindex(closes.index)
        mkt = np.log(b.pct_change().rolling(21).std() * np.sqrt(TRADING_DAYS))
        data["mkt_lrv_21"] = mkt.reindex(data.index.get_level_values("date")).to_numpy()
    return data


HAR_FEATURES = ["lrv_5", "lrv_21", "lrv_63"]


def vol_feature_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in {"target", "label_end"}]


def evaluate_vol(pred_log: pd.Series, data: pd.DataFrame) -> dict[str, float]:
    """Accuracy metrics for log-vol forecasts on rows with a known target."""
    d = data.assign(pred=pred_log).dropna(subset=["pred", "target"])
    err = d["pred"] - d["target"]
    ss_res = float((err**2).sum())
    ss_tot = float(((d["target"] - d["target"].mean()) ** 2).sum())
    vol_err = (np.exp(d["pred"]) - np.exp(d["target"])).abs()
    g = d.groupby(level="date")
    above = g.apply(
        lambda x: float(
            ((x["pred"] > x["pred"].median()) == (x["target"] > x["target"].median())).mean()
        )
    )
    rises = g.apply(
        lambda x: float(((x["pred"] > x["lrv_21"]) == (x["target"] > x["lrv_21"])).mean())
    )
    top = g.apply(
        lambda x: float(
            (
                x.loc[x["pred"] >= x["pred"].quantile(0.8), "target"] >= x["target"].quantile(0.8)
            ).mean()
        )
    )
    return {
        "r2_log": 1.0 - ss_res / ss_tot,
        "mae_vol_points": float(vol_err.mean() * 100),
        "median_abs_pct_error": float((vol_err / np.exp(d["target"])).median()),
        "acc_above_median": float(above.mean()),
        # months are the independent unit: t-stat of monthly accuracy against a coin flip
        "acc_above_median_t": float(
            (above.mean() - 0.5) / (above.std(ddof=1) / np.sqrt(len(above)))
        ),
        "acc_above_median_min_month": float(above.min()),
        "acc_vol_up_or_down": float(rises.mean()),
        "top20_precision": float(top.mean()),
    }
