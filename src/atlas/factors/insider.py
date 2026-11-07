"""Opportunistic insider cluster buying (pre-registered:
research/preregistered/2026-11-07-insider-buying-v1.md).

Signal on day t = number of distinct non-routine insiders with open-market purchases filed in
the 60 calendar days up to and including t-1. A filing is never used before the next trading day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_DAYS = 60
MIN_VALUE = 10_000.0


def routine_purchase_mask(tx: pd.DataFrame) -> pd.Series:
    """True for each row of `tx` that is a ROUTINE trade: the same insider at the same company
    traded (code P or S) in the same calendar month in each of the three previous years.
    `tx` holds P and S rows with owner_cik, issuer_cik and trans_date."""
    keys = set(
        zip(tx["owner_cik"], tx["issuer_cik"], tx["trans_date"].dt.year, tx["trans_date"].dt.month,
            strict=True)
    )  # fmt: skip
    own, iss = tx["owner_cik"].to_numpy(), tx["issuer_cik"].to_numpy()
    y, m = tx["trans_date"].dt.year.to_numpy(), tx["trans_date"].dt.month.to_numpy()
    return pd.Series(
        [all((own[i], iss[i], y[i] - k, m[i]) in keys for k in (1, 2, 3)) for i in range(len(tx))],
        index=tx.index,
    )


def qualifying_purchases(tx: pd.DataFrame) -> pd.DataFrame:
    """Non-routine open-market purchases by officers or directors worth at least MIN_VALUE."""
    routine = routine_purchase_mask(tx)
    buy = tx["code"].eq("P") & ~routine
    insider = tx["relationship"].str.contains("Officer|Director", case=False, regex=True)
    return tx[buy & insider & (tx["value"] >= MIN_VALUE)]


def cluster_signal(
    purchases: pd.DataFrame, dates: pd.DatetimeIndex, columns: pd.Index, window: int = WINDOW_DAYS
) -> pd.DataFrame:
    """Distinct buying insiders active on each trading day. `dates` are the trading days
    (any timezone); an insider counts from the day after their filing for `window` days."""
    cal = (dates + pd.Timedelta(hours=12)).tz_convert("UTC").tz_localize(None).normalize()
    col = {c: i for i, c in enumerate(columns)}
    diff = np.zeros((len(dates) + 1, len(columns)))
    p = purchases[purchases["symbol"].map(lambda s: "EQ:" + s).isin(col)]
    for (sym, _), g in p.groupby(["symbol", "owner_cik"]):
        j = col["EQ:" + sym]
        starts = np.sort(g["filing_date"].dt.normalize().unique()) + np.timedelta64(1, "D")
        s0, e0 = None, None
        for s in starts:  # merge overlapping windows so one insider counts once
            e = s + np.timedelta64(window - 1, "D")
            if s0 is None:
                s0, e0 = s, e
            elif s <= e0 + np.timedelta64(1, "D"):
                e0 = max(e0, e)
            else:
                _add(diff, cal, j, s0, e0)
                s0, e0 = s, e
        if s0 is not None:
            _add(diff, cal, j, s0, e0)
    return pd.DataFrame(np.cumsum(diff[:-1], axis=0), index=dates, columns=columns)


def _add(diff: np.ndarray, cal: pd.DatetimeIndex, j: int, start, end) -> None:
    a = int(cal.searchsorted(pd.Timestamp(start)))
    b = int(cal.searchsorted(pd.Timestamp(end), side="right"))
    if a < len(cal):
        diff[a, j] += 1
        diff[min(b, len(cal)), j] -= 1
