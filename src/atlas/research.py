"""Shared loader for S&P 500 research: prices, universe membership, sectors and factors.

Every research script loads data through here, so validation runs and holdout tests
always see exactly the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from atlas.factors.fundamental import FUNDAMENTAL_FACTORS, fundamental_factor_frames
from atlas.factors.panel import price_factor_frames
from atlas.marketdata.bars import Adjustment
from atlas.marketdata.loader import close_panel, load_for_backtest
from atlas.marketdata.sec import items_for, load_facts
from atlas.marketdata.store import BarStore
from atlas.marketdata.universe import membership_from_additions, sp500_addition_dates
from atlas.refdata.instruments import SecurityMaster

CONSTITUENTS = "data/universes/sp500_constituents.csv"
UNIVERSE = "data/universes/sp500.txt"


@dataclass
class ResearchData:
    bars: dict[str, pd.DataFrame]
    secmaster: SecurityMaster
    closes: pd.DataFrame  # total-return adjusted closes (for returns)
    split_closes: pd.DataFrame  # split-adjusted only: the prices shown on charts
    membership: pd.DataFrame
    benchmark: pd.Series  # SPY adjusted close
    sectors: dict[str, str]
    factors: dict[str, pd.DataFrame]


def load_research_data(features: str = "all", data_dir: str = "data") -> ResearchData:
    """features: "price", "fundamental" or "all"."""
    store = BarStore(data_dir)
    sm = SecurityMaster()
    symbols = Path(UNIVERSE).read_text().split()
    bars = load_for_backtest(store, symbols, sm)
    closes = close_panel(bars)
    members = membership_from_additions(sp500_addition_dates(CONSTITUENTS), closes.index)
    cons = pd.read_csv(CONSTITUENTS)
    sectors = dict(
        zip(
            "EQ:" + cons["Symbol"].str.replace("-", ".", regex=False),
            cons["GICS Sector"],
            strict=True,
        )
    )
    present = [s for s in symbols if f"EQ:{s}" in closes.columns]

    def panel(adj: Adjustment) -> pd.DataFrame:
        return pd.DataFrame(
            {f"EQ:{s}": store.read(s, adjustment=adj)["close"] for s in present}
        ).reindex(closes.index)

    split = panel(Adjustment.SPLIT)
    factors: dict[str, pd.DataFrame] = {}
    if features in ("price", "all"):
        factors.update(price_factor_frames(closes))
    if features in ("fundamental", "all"):
        fund = fundamental_factor_frames(
            items_for(load_facts(), present), split, panel(Adjustment.RAW)
        )
        factors.update({k: fund[k] for k in FUNDAMENTAL_FACTORS if k in fund})
    return ResearchData(
        bars, sm, closes, split, members, store.read("SPY")["close"], sectors, factors
    )
