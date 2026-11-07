"""Candidate signal: insider_cluster_buying.

Written BEFORE testing (research/preregistered/2026-11-07-insider-buying-v1.md); never edited
after its holdout opens (a change is a new trial under a new name).
"""

from __future__ import annotations

import pandas as pd

from atlas.factors.insider import cluster_signal, qualifying_purchases
from atlas.lab.evaluate import CandidateSpec
from atlas.marketdata.insider import load_transactions
from atlas.research import ResearchData

SPEC = CandidateSpec(
    name="insider_cluster_buying",
    hypothesis="Stocks where several corporate insiders (officers and directors) recently "
    "bought shares on the open market, outside their usual trading pattern, outperform over "
    "the next month",
    rationale="Insiders know their business better than outsiders. Lakonishok & Lee (2001) and "
    "Cohen, Malloy & Pomorski (2012, J. Finance) find that insider purchases predict returns, "
    "and that only 'opportunistic' trades (not the same calendar month each year) carry "
    "information. Several insiders buying together is a stronger sign than one. Filings are "
    "public within two business days and are used from the next trading day on",
    horizon_days=21,
    author="claude",
    notes="Source: SEC Form 4 quarterly data sets 2013 on. Routine rule simplified: purchases "
    "in the same month in each of the previous three years are dropped; first-time buyers are "
    "kept. The effect is reported to be concentrated in smaller firms and may have weakened "
    "since 2012. Dev 2017-2024, holdout 2025 on. Amendments (4/A) are ignored",
)


def compute(data: ResearchData) -> pd.DataFrame:
    purchases = qualifying_purchases(load_transactions())
    return cluster_signal(purchases, data.closes.index, data.closes.columns)
