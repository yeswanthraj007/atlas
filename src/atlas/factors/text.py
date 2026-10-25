"""Filing-text change signals ("Lazy Prices", Cohen, Malloy & Nguyen 2020, J. Finance).

Each filing is compared with the comparable filing a year earlier (10-K vs last year's
10-K; 10-Q vs the same quarter's 10-Q last year). Companies whose reports barely change
("non-changers") have historically outperformed those that change them a lot, with the
effect building over several months because investors do not read the changes.

Signals (HIGHER = more similar = more attractive, the sign fixed by the paper, not fitted):
  * text_sim_cosine   cosine similarity of word-count vectors (whole document)
  * text_sim_jaccard  Jaccard similarity of the sets of words used
  * rf_sim_cosine     cosine similarity of the Risk Factors section (Item 1A)
  * text_len_stable   minus |log(length now / length a year ago)|

Point-in-time: usable from the first trading day after EDGAR's acceptance date.
No language model is involved, so there is no memorised-future contamination.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.marketdata.filings import read_text, risk_factors_section

_WORD = re.compile(r"[a-z]{2,}")


def word_counts(text: str) -> Counter[str]:
    return Counter(_WORD.findall(text.lower()))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return float("nan")
    small, big = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * big.get(k, 0) for k, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def jaccard(a: Counter[str], b: Counter[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def match_previous(index: pd.DataFrame, tolerance_days: int = 45) -> pd.DataFrame:
    """For every filing, the same company's same-form filing covering the period one year
    earlier (within `tolerance_days`). Adds column `prev_accession` (None if none)."""
    df = index.copy().reset_index(drop=True)
    df["period_ts"] = pd.to_datetime(df["period"], errors="coerce")
    df["prev_accession"] = None
    for _, g in df.dropna(subset=["period_ts"]).groupby(["symbol", "form"]):
        periods = g["period_ts"]
        for label, p in periods.items():
            gap = (periods - (p - pd.Timedelta(days=365))).abs().dt.days
            j = gap.idxmin()
            if gap[j] <= tolerance_days and j != label:
                df.at[label, "prev_accession"] = g.at[j, "accession"]
    return df.drop(columns="period_ts").sort_values(["symbol", "form", "period"])


def compute_similarities(
    index: pd.DataFrame, root: str | Path = "data/sec", progress=print
) -> pd.DataFrame:
    matched = match_previous(index)
    cache: dict[str, tuple[Counter[str], Counter[str], int]] = {}

    def load(cik: int, acc: str) -> tuple[Counter[str], Counter[str], int]:
        if acc not in cache:
            text = read_text(root, cik, acc)
            cache[acc] = (word_counts(text), word_counts(risk_factors_section(text)), len(text))
        return cache[acc]

    rows = []
    for n, r in enumerate(matched.itertuples(), 1):
        if not isinstance(r.prev_accession, str):
            continue
        try:
            cur, cur_rf, cur_len = load(r.cik, r.accession)
            old, old_rf, old_len = load(r.cik, r.prev_accession)
        except FileNotFoundError:
            continue
        rows.append(
            {
                "symbol": r.symbol,
                "form": r.form,
                "accession": r.accession,
                "accepted": r.accepted,
                "period": r.period,
                "text_sim_cosine": cosine(cur, old),
                "text_sim_jaccard": jaccard(cur, old),
                "rf_sim_cosine": cosine(cur_rf, old_rf) if cur_rf and old_rf else np.nan,
                "text_len_stable": -abs(math.log(max(cur_len, 1) / max(old_len, 1))),
            }
        )
        if n % 2000 == 0:
            progress(f"  compared {n}/{len(matched)}")
        if len(cache) > 4000:  # bound memory: groups are processed company by company
            cache.clear()
    return pd.DataFrame(rows)


TEXT_FACTORS = ["text_sim_cosine", "text_sim_jaccard", "rf_sim_cosine", "text_len_stable"]


def text_factor_frames(
    sims: pd.DataFrame, dates: pd.DatetimeIndex, prefix: str = "EQ:", max_age_days: int = 400
) -> dict[str, pd.DataFrame]:
    """Daily point-in-time frames: each date sees the latest filing accepted BEFORE it
    (usable from the next trading day), for at most `max_age_days`."""
    s = sims.copy()
    accepted = pd.to_datetime(s["accepted"], utc=True)
    pos = dates.searchsorted(accepted, side="right")  # first trading date after acceptance
    pos_ok = pos < len(dates)
    s = s[pos_ok]
    s["usable"] = dates[pos[pos_ok]]
    out: dict[str, pd.DataFrame] = {}
    for name in TEXT_FACTORS:
        cols = {}
        for sym, g in s.groupby("symbol"):
            g = g.dropna(subset=[name]).sort_values("usable").drop_duplicates("usable", keep="last")
            if g.empty:
                continue
            ser = pd.Series(g[name].to_numpy(), index=pd.DatetimeIndex(g["usable"]))
            daily = ser.reindex(dates).ffill()
            age = pd.Series(pd.DatetimeIndex(g["usable"]), index=g["usable"].to_numpy())
            last_usable = age.reindex(dates).ffill()
            stale = (dates - pd.DatetimeIndex(last_usable)).days > max_age_days
            cols[prefix + sym] = daily.where(~stale)
        out[name] = pd.DataFrame(cols, index=dates)
    return out
