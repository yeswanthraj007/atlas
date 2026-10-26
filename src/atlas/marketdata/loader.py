"""Bridge from the bar store to the rest of the platform (backtests, risk, research)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from atlas.marketdata.bars import Adjustment, Timeframe
from atlas.marketdata.quality import (
    QualityConfig,
    check_bars,
    has_errors,
    trim_synthetic_history,
)
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
    trim_synthetic: bool = True,
    skip_broken: bool = False,
) -> dict[str, pd.DataFrame]:
    """Adjusted daily bars keyed by instrument id, registering instruments as needed.

    With `strict`, any symbol with ERROR-level quality issues raises instead of being
    silently included: a backtest should never run on data known to be broken.
    With `trim_synthetic`, history before a vendor-filled dormant period is dropped, so a
    reused ticker only carries the current company's prices.
    With `skip_broken`, symbols with errors or no stored data are left out (and listed in
    `load_for_backtest.skipped`) instead of raising: for research over large universes.
    """
    etfs = etfs or set()
    out: dict[str, pd.DataFrame] = {}
    broken: dict[str, list[str]] = {}
    skipped: dict[str, list[str]] = {}
    for sym in symbols:
        try:
            df = store.read(sym, Timeframe.DAY, Adjustment.ALL, start, end)
        except FileNotFoundError:
            skipped[sym] = ["no data"]
            continue
        if trim_synthetic:
            df = trim_synthetic_history(df)
        issues = check_bars(sym, df, QualityConfig())
        if has_errors(issues):
            broken[sym] = [i.check for i in issues if i.severity == "error"]
            if strict or skip_broken:
                skipped[sym] = broken[sym]
                continue
        inst: Instrument = ETF.of(sym) if sym in etfs else Equity.of(sym)
        inst = secmaster.add(inst)
        out[inst.id] = df
    load_for_backtest.skipped = skipped  # type: ignore[attr-defined]
    if broken and strict and not skip_broken:
        raise ValueError(f"data quality errors, refusing to backtest: {broken}")
    return out


def close_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({k: v["close"] for k, v in bars.items()}).sort_index()


def returns_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return close_panel(bars).pct_change().iloc[1:]
