"""Supervised dataset for the cross-sectional return model.

One row per (rebalance date, stock in the universe that day):
  * features: each factor's cross-sectional percentile rank that day, centred on 0
    (-0.5 worst .. +0.5 best). Ranking removes scale and drift over time, which is what
    makes a model trained on 2017 usable in 2024. Missing values stay NaN.
  * context features: market regime on that date (benchmark volatility and trend), the
    same for every stock, so the model can learn when each factor works.
  * target: the stock's cross-sectional rank of its next-`horizon`-day return, centred.
  * `label_end`: the date the target becomes known. Training may only use rows whose
    label_end is before the training cutoff (no peeking at outcomes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def centred_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True) - 0.5


def regime_features(benchmark: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    b = benchmark.sort_index()
    r = b.pct_change()
    out = pd.DataFrame(
        {
            "mkt_vol_21d": r.rolling(21).std() * np.sqrt(252),
            "mkt_vol_ratio": r.rolling(21).std() / r.rolling(252).std(),
            "mkt_trend_200d": b / b.rolling(200).mean() - 1.0,
            "mkt_ret_63d": b / b.shift(63) - 1.0,
        }
    )
    return out.reindex(out.index.union(dates)).ffill().reindex(dates)


def build_dataset(
    factors: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
    horizon: int = 21,
    membership: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Long table indexed by (date, instrument) with features, `target`, `label_end`."""
    closes = closes.sort_index()
    pos = closes.index.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("every sample date must be a trading date in `closes`")
    end_pos = pos + horizon
    valid_label = end_pos < len(closes)
    label_end = pd.Series(
        [closes.index[p] if ok else pd.NaT for p, ok in zip(end_pos, valid_label, strict=True)],
        index=dates,
    )
    fwd = pd.DataFrame(
        {
            d: (closes.iloc[e] / closes.iloc[p] - 1.0) if ok else np.nan
            for d, p, e, ok in zip(dates, pos, end_pos, valid_label, strict=True)
        }
    ).T
    mask = None
    if membership is not None:
        mask = membership.reindex(index=dates, columns=closes.columns).fillna(False).astype(bool)

    def ranked(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.reindex(index=dates, columns=closes.columns)
        if mask is not None:
            f = f.where(mask)
        return centred_rank(f)

    feats = {name: ranked(f).stack(future_stack=True) for name, f in factors.items()}
    data = pd.DataFrame(feats)
    universe = closes.reindex(dates).notna()
    if mask is not None:
        universe &= mask
    keep = universe.stack(future_stack=True)
    data = data[keep.reindex(data.index).fillna(False).to_numpy()]
    data = data.dropna(how="all")
    tgt = ranked(fwd).stack(future_stack=True)
    data["target"] = tgt.reindex(data.index)
    data.index.names = ["date", "instrument"]
    data["label_end"] = label_end.reindex(data.index.get_level_values("date")).to_numpy()
    if benchmark is not None:
        reg = regime_features(benchmark, dates)
        data = data.join(reg, on="date")
    return data


FEATURE_EXCLUDE = {"target", "label_end"}


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in FEATURE_EXCLUDE]
