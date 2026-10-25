"""Bridge from the bar store to the rest of the platform (backtests, risk, research)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from atlas.marketdata.bars import Adjustment, Timeframe
from atlas.marketdata.quality import QualityConfig, check_bars, has_errors
from atlas.marketdata.store import BarStore
from atlas.refdata.instruments import ETF, Equity, Instrument, SecurityMaster


def load_for_backtest(
    store: BarStore,
    symbols: list[str],
    secmaster: SecurityMaster,
    start: datetime | None = None,
    end: datetime | None = None,
    etfs: set[str] | None = None,
    strict: bool = True,
) -> dict[str, pd.DataFrame]:
    """Adjusted daily bars keyed by instrument id, registering instruments as needed.

    With `strict`, any symbol with ERROR-level quality issues raises instead of being
    silently included: a backtest should never run on data known to be broken.
    """
    etfs = etfs or set()
    out: dict[str, pd.DataFrame] = {}
    broken: dict[str, list[str]] = {}
    for sym in symbols:
        df = store.read(sym, Timeframe.DAY, Adjustment.ALL, start, end)
        issues = check_bars(sym, df, QualityConfig())
        if has_errors(issues):
            broken[sym] = [i.check for i in issues if i.severity == "error"]
            if strict:
                continue
        inst: Instrument = ETF.of(sym) if sym in etfs else Equity.of(sym)
        inst = secmaster.add(inst)
        out[inst.id] = df
    if broken and strict:
        raise ValueError(f"data quality errors, refusing to backtest: {broken}")
    return out


def close_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({k: v["close"] for k, v in bars.items()}).sort_index()


def returns_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return close_panel(bars).pct_change().iloc[1:]
