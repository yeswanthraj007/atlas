"""Candidate signal: customer_momentum.

Written BEFORE testing; never edited after its holdout opens (a change is a new trial).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from atlas.factors.links import link_matrix
from atlas.lab.evaluate import CandidateSpec
from atlas.research import ResearchData

SPEC = CandidateSpec(
    name="customer_momentum",
    hypothesis="Companies whose customers (named in their own 10-K) had high returns last "
    "month outperform next month",
    rationale="Cohen & Frazzini (2008, JF): attention-constrained investors do not "
    "immediately price news about a firm's customers into the supplier, so customer "
    "returns predict supplier returns. Example of the mechanism: a big customer committing "
    "more volume to a supplier is good news the supplier's price absorbs slowly",
    horizon_days=21,
    author="claude",
    notes="Links extracted by name matching in 10-K sentences with customer words; only "
    "S&P 500 customers are visible, so coverage is partial. Dev through 2023, holdout "
    "2024-2026 at the user's request",
)
LINKS = Path("data/sec/links.parquet")


def compute(data: ResearchData) -> pd.DataFrame:
    links = pd.read_parquet(LINKS)
    past = data.closes / data.closes.shift(21) - 1.0
    rows = {}
    for d in data.closes.index[::5]:
        m = link_matrix(links, d, "customer", min_mentions=2)
        if not m:
            continue
        r_today = past.loc[d]
        sig = {}
        for src, customers in m.items():
            w = np.array([customers[t] for t in customers], dtype=float)
            r = np.array([r_today.get(f"EQ:{t}", np.nan) for t in customers])
            ok = np.isfinite(r)
            if ok.any():
                sig[f"EQ:{src}"] = float((w[ok] * r[ok]).sum() / w[ok].sum())
        rows[d] = pd.Series(sig, dtype=float)
    out = pd.DataFrame(rows).T
    return out.reindex(data.closes.index).ffill(limit=4).reindex(columns=data.closes.columns)
