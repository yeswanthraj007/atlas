"""Canonical bar format and the provider interface.

Every provider returns the same shape, so storage, quality checks and backtests never
care where data came from:

    DataFrame indexed by a tz-aware UTC DatetimeIndex named "timestamp", sorted, unique,
    with float columns  open, high, low, close, volume  (plus optional vwap, trade_count).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

import pandas as pd

REQUIRED = ["open", "high", "low", "close", "volume"]
OPTIONAL = ["vwap", "trade_count"]


class Timeframe(StrEnum):
    MINUTE = "1min"
    FIVE_MINUTE = "5min"
    HOUR = "1h"
    DAY = "1d"


class Adjustment(StrEnum):
    """RAW = as traded. ALL = split- and dividend-adjusted (total-return prices).

    Research and backtests use ALL so returns include dividends and survive splits.
    RAW is kept for reconciliation with broker fills and for corporate-action audits.
    Adjusted history changes every time a split or dividend happens, so ALL datasets
    are re-downloaded whole rather than appended to.
    """

    RAW = "raw"
    ALL = "all"


class BarProvider(Protocol):
    name: str

    def get_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = Timeframe.DAY,
        adjustment: Adjustment = Adjustment.ALL,
    ) -> dict[str, pd.DataFrame]: ...


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any provider's frame into the canonical format (raises if impossible)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"bars missing columns {missing}")
    out = df[[c for c in REQUIRED + OPTIONAL if c in df.columns]].astype("float64")
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx.rename("timestamp")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
