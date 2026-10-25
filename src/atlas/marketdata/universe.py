"""Point-in-time investable universes.

A backtest must only pick from stocks that were actually in the universe on each date.
Selecting from today's index members silently buys future winners: many stocks join an
index *because* their price already rose.

`membership_from_additions` uses each member's index-addition date. It removes inclusion
bias but cannot add back companies that were later removed from the index (bankruptcies,
acquisitions, demotions); a full point-in-time constituent history is needed for that.
"""

from __future__ import annotations

import pandas as pd


def membership_from_additions(
    added: dict[str, str | pd.Timestamp], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Boolean frame (dates x keys): True from each key's addition date onward."""
    added_ts = {k: pd.Timestamp(v) for k, v in added.items()}
    tz = dates.tz
    cols = {}
    for key, ts in added_ts.items():
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        cols[key] = dates >= ts
    return pd.DataFrame(cols, index=dates)


def sp500_addition_dates(constituents_csv: str, prefix: str = "EQ:") -> dict[str, str]:
    """Map instrument id -> 'Date added' from a constituents CSV (Symbol, Date added)."""
    df = pd.read_csv(constituents_csv)
    sym = df["Symbol"].astype(str).str.strip().str.replace("-", ".", regex=False)
    return dict(zip(prefix + sym, df["Date added"].astype(str).str.strip(), strict=True))
