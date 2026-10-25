"""Price-based equity factors. Every factor is oriented so that HIGHER = MORE ATTRACTIVE.

Each function takes a window of daily closes (rows = dates, oldest first; columns =
instruments) ending at the decision date, and returns one score per instrument that has
complete data over the factor's lookback. Nothing after the last row is ever used.

| factor          | idea                                               | reference                 |
| --------------- | -------------------------------------------------- | ------------------------- |
| mom_12_1        | 12-month return, skipping the last month           | Jegadeesh & Titman 1993   |
| vol_adj_mom     | mom_12_1 divided by volatility (trend quality)     | Barroso & Santa-Clara 2015|
| reversal_1m     | minus last month's return (short-term reversal)    | Jegadeesh 1990            |
| low_vol         | minus 1-year volatility (low-volatility anomaly)   | Ang et al. 2006           |
| high_52w        | price relative to its 52-week high                 | George & Hwang 2004       |
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

YEAR, MONTH = 252, 21

FactorFn = Callable[[pd.DataFrame], pd.Series]


def _complete(closes: pd.DataFrame, rows: int) -> pd.DataFrame:
    window = closes.tail(rows)
    if len(window) < rows:
        return window.iloc[:, :0]
    return window.dropna(axis=1, how="any")


def mom_12_1(closes: pd.DataFrame) -> pd.Series:
    w = _complete(closes, YEAR + 1)
    if w.empty:
        return pd.Series(dtype=float)
    return w.iloc[-1 - MONTH] / w.iloc[0] - 1.0


def _ann_vol(w: pd.DataFrame) -> pd.Series:
    return w.pct_change().iloc[1:].std(ddof=1) * np.sqrt(YEAR)


def vol_adj_mom(closes: pd.DataFrame) -> pd.Series:
    w = _complete(closes, YEAR + 1)
    if w.empty:
        return pd.Series(dtype=float)
    vol = _ann_vol(w).replace(0.0, np.nan)
    return ((w.iloc[-1 - MONTH] / w.iloc[0] - 1.0) / vol).dropna()


def reversal_1m(closes: pd.DataFrame) -> pd.Series:
    w = _complete(closes, MONTH + 1)
    if w.empty:
        return pd.Series(dtype=float)
    return -(w.iloc[-1] / w.iloc[0] - 1.0)


def low_vol(closes: pd.DataFrame) -> pd.Series:
    w = _complete(closes, YEAR + 1)
    if w.empty:
        return pd.Series(dtype=float)
    return -_ann_vol(w)


def high_52w(closes: pd.DataFrame) -> pd.Series:
    w = _complete(closes, YEAR)
    if w.empty:
        return pd.Series(dtype=float)
    return w.iloc[-1] / w.max()


PRICE_FACTORS: dict[str, FactorFn] = {
    "mom_12_1": mom_12_1,
    "vol_adj_mom": vol_adj_mom,
    "reversal_1m": reversal_1m,
    "low_vol": low_vol,
    "high_52w": high_52w,
}
