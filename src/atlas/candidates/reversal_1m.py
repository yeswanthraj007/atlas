"""Candidate signal: reversal_1m.

Written BEFORE testing; never edited after its holdout opens (a change is a new trial).
"""

from __future__ import annotations

import pandas as pd

from atlas.factors.panel import price_factor_frames
from atlas.lab.evaluate import CandidateSpec
from atlas.research import ResearchData

SPEC = CandidateSpec(
    name="reversal_1m",
    hypothesis="Last month's biggest losers outperform last month's biggest winners over the "
    "next month, mostly among mid and small caps, but possibly not after trading costs",
    rationale="Short-term reversal compensates liquidity providers who absorb temporary order "
    "imbalances (Jegadeesh 1990; Nagel 2012). That makes it strongest exactly where trading "
    "is expensive, so the after-cost gate is the real test",
    horizon_days=21,
    author="claude",
    notes="Universe built from today's index members (survivorship bias); the S&P 500-only "
    "version of this factor was tested earlier (backfilled trials)",
)


def compute(data: ResearchData) -> pd.DataFrame:
    return price_factor_frames(data.closes)["reversal_1m"]
