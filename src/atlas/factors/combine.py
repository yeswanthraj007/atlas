"""Turn raw factor values into comparable scores and combine them.

Pipeline per rebalance date (all cross-sectional, so only same-day information is used):
  1. winsorize: clip each factor at its 2.5th / 97.5th percentiles, so one bad print or
     one extreme stock cannot dominate;
  2. z-score: subtract the mean, divide by the standard deviation;
  3. sector-neutralise (optional): subtract each sector's mean z-score, so the composite
     ranks stocks against their peers instead of betting on whole sectors;
  4. combine: weighted sum of factor z-scores. Weights are fixed, or set from each
     factor's trailing information coefficient (only past, completed periods).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats


def winsorize(s: pd.Series, lower: float = 0.025, upper: float = 0.975) -> pd.Series:
    if s.empty:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if s.empty or not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def sector_neutral(z: pd.Series, sectors: Mapping[str, str] | None) -> pd.Series:
    if not sectors or z.empty:
        return z
    groups = pd.Series({k: sectors.get(k, "unknown") for k in z.index})
    return z - z.groupby(groups).transform("mean")


def standardize(raw: pd.Series, sectors: Mapping[str, str] | None = None) -> pd.Series:
    s = raw.replace([np.inf, -np.inf], np.nan).dropna()
    return sector_neutral(zscore(winsorize(s)), sectors)


def composite(
    scores: Mapping[str, pd.Series],
    weights: Mapping[str, float],
    min_factors: int = 1,
) -> pd.Series:
    """Weighted average of factor z-scores over the factors each stock has.

    Stocks missing a factor are scored on the remaining ones (weights renormalised), but
    need at least `min_factors` of them.
    """
    frame = pd.DataFrame({k: v for k, v in scores.items() if weights.get(k, 0) > 0})
    if frame.empty:
        return pd.Series(dtype=float)
    w = pd.Series({k: weights[k] for k in frame.columns}, dtype=float)
    present = frame.notna()
    num = frame.fillna(0.0).mul(w, axis=1).sum(axis=1)
    den = present.mul(w, axis=1).sum(axis=1)
    out = num / den.replace(0.0, np.nan)
    need = min(min_factors, frame.shape[1])  # zero-weight factors don't count
    return out[present.sum(axis=1) >= need].dropna()


def ic_weights(
    ic_history: pd.DataFrame, min_periods: int = 6, floor: float = 0.0
) -> dict[str, float]:
    """Weights proportional to each factor's mean past IC, floored at `floor`.

    A factor that has not predicted returns gets zero weight instead of a negative one:
    flipping a factor's sign because of a short noisy history is a classic overfit.
    Falls back to equal weights until there is enough history.
    """
    cols = list(ic_history.columns)
    hist = ic_history.dropna(how="all")
    if len(hist) < min_periods:
        return dict.fromkeys(cols, 1.0)
    mean_ic = hist.mean().clip(lower=floor)
    if mean_ic.sum() <= 0:
        return dict.fromkeys(cols, 1.0)
    return (mean_ic / mean_ic.sum()).to_dict()


def spearman_ic(score: pd.Series, fwd_return: pd.Series) -> float:
    both = pd.concat([score, fwd_return], axis=1, keys=["s", "r"]).dropna()
    if len(both) < 10 or both["s"].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(both["s"], both["r"]).statistic)
