"""Data quality checks. Bad data is the most common reason a backtest lies.

Each check returns issues instead of silently fixing data: whoever consumes the data
decides whether an issue is fatal. `ERROR` issues mean the series must not be used
until fixed; `WARN` issues need a human look.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from atlas.marketdata.bars import REQUIRED


class Severity(StrEnum):
    ERROR = "error"
    WARN = "warn"


@dataclass(frozen=True)
class Issue:
    symbol: str
    check: str
    severity: Severity
    count: int
    first: pd.Timestamp | None
    detail: str


@dataclass(frozen=True)
class QualityConfig:
    max_abs_return: float = 0.40  # a 40%+ daily move is usually a missed split or a bad tick
    stale_run: int = 5  # identical closes on this many consecutive bars
    max_gap_days: int = 5  # calendar days between daily bars before we call it a gap
    min_rows: int = 20


def _issue(symbol: str, check: str, sev: Severity, mask: pd.Series, detail: str) -> Issue | None:
    n = int(mask.sum())
    if n == 0:
        return None
    return Issue(symbol, check, sev, n, mask.index[mask.to_numpy()][0], detail)


def check_bars(
    symbol: str, df: pd.DataFrame, cfg: QualityConfig | None = None, daily: bool = True
) -> list[Issue]:
    cfg = cfg or QualityConfig()
    out: list[Issue | None] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [Issue(symbol, "schema", Severity.ERROR, len(missing), None, f"missing {missing}")]
    if len(df) < cfg.min_rows:
        out.append(Issue(symbol, "too_short", Severity.WARN, len(df), None, f"only {len(df)} rows"))
    if len(df) == 0:
        return [i for i in out if i]

    o, h, lo, c, v = (df[k] for k in REQUIRED)
    out.append(
        _issue(symbol, "nan", Severity.ERROR, df[REQUIRED].isna().any(axis=1), "NaN in OHLCV")
    )
    out.append(
        _issue(
            symbol,
            "non_positive_price",
            Severity.ERROR,
            (df[["open", "high", "low", "close"]] <= 0).any(axis=1),
            "price <= 0",
        )
    )
    out.append(_issue(symbol, "negative_volume", Severity.ERROR, v < 0, "volume < 0"))
    out.append(
        _issue(
            symbol,
            "ohlc_inconsistent",
            Severity.ERROR,
            (h < np.maximum(o, c) - 1e-9) | (lo > np.minimum(o, c) + 1e-9) | (h < lo),
            "high/low do not contain open/close",
        )
    )
    out.append(
        _issue(
            symbol,
            "duplicate_timestamp",
            Severity.ERROR,
            pd.Series(df.index.duplicated(), index=df.index),
            "duplicate bars",
        )
    )
    if not df.index.is_monotonic_increasing:
        out.append(Issue(symbol, "unsorted", Severity.ERROR, 1, None, "index not sorted"))

    ret = c.pct_change()
    out.append(
        _issue(
            symbol,
            "extreme_return",
            Severity.WARN,
            ret.abs() > cfg.max_abs_return,
            f"|close-to-close return| > {cfg.max_abs_return:.0%}",
        )
    )
    same = c.diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    out.append(
        _issue(
            symbol,
            "stale_price",
            Severity.WARN,
            run >= cfg.stale_run,
            f"close unchanged for {cfg.stale_run}+ bars",
        )
    )
    out.append(_issue(symbol, "zero_volume", Severity.WARN, v == 0, "zero-volume bars"))
    if daily and len(df) > 1:
        gaps = pd.Series(df.index, index=df.index).diff().dt.days
        out.append(
            _issue(
                symbol,
                "gap",
                Severity.WARN,
                gaps > cfg.max_gap_days,
                f"more than {cfg.max_gap_days} calendar days between bars",
            )
        )
    return [i for i in out if i is not None]


def has_errors(issues: list[Issue]) -> bool:
    return any(i.severity is Severity.ERROR for i in issues)


def report(issues: list[Issue]) -> pd.DataFrame:
    return (
        pd.DataFrame([i.__dict__ for i in issues])
        if issues
        else pd.DataFrame(columns=["symbol", "check", "severity", "count", "first", "detail"])
    )
