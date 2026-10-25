"""Candidate signal: tech_peer_catchup.

Written BEFORE testing; never edited after its holdout opens (a change is a new trial).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas.factors.peers import build_fingerprints, peer_momentum
from atlas.lab.evaluate import CandidateSpec
from atlas.marketdata.filings import load_index
from atlas.ml.text_embed import ENCODER_REVISION
from atlas.research import ResearchData

SPEC = CandidateSpec(
    name="tech_peer_catchup",
    hypothesis="Stocks whose closest technology peers (by 10-K business description) rose "
    "more than they did over the past month outperform over the next month",
    rationale="Attention-constrained investors are slow to carry news across related firms: "
    "technology-linked returns predict focal returns (Lee, Sun, Wang & Zhang 2019, JFE; "
    "117 bps/month alpha) and customer shocks predict supplier returns (Cohen & Frazzini "
    "2008, JF). A theme lifting its leaders (e.g. AI data-centre demand) should reach the "
    "lagging firms in the same technology space with a delay",
    horizon_days=21,
    author="claude",
    notes="Peers from a text encoder pinned to 2024-11-05 (all-mpnet-base-v2); evaluated "
    "with development through 2023 and holdout 2024-2026 at the user's request",
)
CACHE = Path(f"data/sec/business_fingerprints_{ENCODER_REVISION[:10]}.npz")


def compute(data: ResearchData) -> pd.DataFrame:
    fp = build_fingerprints(load_index(), CACHE)
    dates = data.closes.index[::5]  # weekly; a one-month signal does not need daily updates
    sig = peer_momentum(fp, data.closes, dates, lookback=21, k=10)
    return sig.reindex(data.closes.index).ffill(limit=4).reindex(columns=data.closes.columns)
