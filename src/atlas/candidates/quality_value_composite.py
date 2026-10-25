"""Candidate signal: quality_value_composite.

Written BEFORE testing; never edited after its holdout opens (a change is a new trial).
"""

from __future__ import annotations

import pandas as pd

from atlas.factors.combine import standardize
from atlas.lab.evaluate import CandidateSpec
from atlas.research import ResearchData

SPEC = CandidateSpec(
    name="quality_value_composite",
    hypothesis="Profitable, conservatively accounted companies that are cheap relative to "
    "earnings and book value outperform the S&P 500 median over the next three months",
    rationale="Quality (gross profitability, low accruals) and value are among the few factor "
    "themes that replicate out of sample (Jensen, Kelly & Pedersen 2023) and, with "
    "quarterly turnover, survive trading costs (Novy-Marx & Velikov). Investors overpay "
    "for exciting growth and underweight dull, profitable firms; combining four related "
    "measures reduces the noise of any single accounting ratio",
    horizon_days=63,
    author="claude",
    notes="Single factors in this composite were looked at individually on 2025-2026 in "
    "backfilled trials (all |t| < 1.2); the composite itself is new",
)

COMPONENTS = ("gross_profitability", "low_accruals", "earnings_yield", "book_to_price")


def compute(data: ResearchData) -> pd.DataFrame:
    """Average of cross-sectional z-scores (winsorized), computed date by date on the
    stocks that have each measure; needs at least three of the four."""
    frames = {k: data.factors[k] for k in COMPONENTS if k in data.factors}
    dates = data.closes.index
    rows = {}
    for d in dates[::5]:  # weekly is plenty for a 3-month signal; keeps evaluation fast
        zs = [standardize(f.loc[d].dropna()) for f in frames.values() if d in f.index]
        if not zs:
            continue
        panel = pd.concat(zs, axis=1)
        rows[d] = panel.mean(axis=1)[panel.notna().sum(axis=1) >= 3]
    out = pd.DataFrame(rows).T.reindex(dates).ffill(limit=4)
    return out.reindex(columns=data.closes.columns)
