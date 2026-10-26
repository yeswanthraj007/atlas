"""Candidate signal: momentum_12_1.

Written BEFORE testing; never edited after its holdout opens (a change is a new trial).
"""

from __future__ import annotations

import pandas as pd

from atlas.factors.panel import price_factor_frames
from atlas.lab.evaluate import CandidateSpec
from atlas.research import ResearchData

SPEC = CandidateSpec(
    name="momentum_12_1",
    hypothesis="Stocks with the highest 12-month returns, skipping the latest month, "
    "outperform over the next month, more so among mid and small caps",
    rationale="Investors underreact to gradual information and herd late (Jegadeesh & Titman "
    "1993); the effect is weaker where analyst coverage is dense, so it should be stronger "
    "outside the S&P 500. Monthly rebalancing of a slow signal keeps turnover moderate",
    horizon_days=21,
    author="claude",
    notes="Universe built from today's index members (survivorship bias); the S&P 500-only "
    "version of this factor was tested earlier (backfilled trials)",
)


def compute(data: ResearchData) -> pd.DataFrame:
    return price_factor_frames(data.closes)["mom_12_1"]
