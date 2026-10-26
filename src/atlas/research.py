"""Shared loader for S&P 500 research: prices, universe membership, sectors and factors.

Every research script loads data through here, so validation runs and holdout tests
always see exactly the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from atlas.factors.earnings import earnings_factor_frames, volume_factor_frames
from atlas.factors.fundamental import FUNDAMENTAL_FACTORS, fundamental_factor_frames
from atlas.factors.panel import price_factor_frames
from atlas.marketdata.bars import Adjustment
from atlas.marketdata.loader import close_panel, load_for_backtest
from atlas.marketdata.sec import items_for, load_facts
from atlas.marketdata.store import BarStore
from atlas.marketdata.universe import membership_from_additions, sp500_addition_dates
from atlas.refdata.instruments import SecurityMaster
from atlas.risk.liquidity import corwin_schultz_spread

CONSTITUENTS = "data/universes/sp500_constituents.csv"
UNIVERSE = "data/universes/sp500.txt"
UNIVERSES = {
    "sp500": "data/universes/sp500.txt",
    "midsmall": "data/universes/midsmall.txt",
    "sp1500": "data/universes/sp1500.txt",
}
ALL_CONSTITUENTS = [
    "data/universes/sp500_constituents.csv",
    "data/universes/sp400_constituents.csv",
    "data/universes/sp600_constituents.csv",
]


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
    spread: pd.DataFrame | None = None  # Corwin-Schultz round-trip cost estimate per stock-day


def load_research_data(
    features: str = "all", data_dir: str = "data", universe: str = "sp500"
) -> ResearchData:
    """features: "price", "fundamental", "all", or "extended" (all + earnings-event and
    volume features). universe: "sp500", "midsmall" (S&P 400 + 600) or "sp1500".

    Membership: S&P 500 names enter on their index-addition date. Mid/small names have no
    addition dates in the free source, so they count from their first price (today's
    members only: survivorship bias, disclosed in every result on these universes).
    `spread` holds the Corwin-Schultz trading-cost estimate (not a model feature)."""
    store = BarStore(data_dir)
    sm = SecurityMaster()
    symbols = Path(UNIVERSES[universe]).read_text().split()
    bars = load_for_backtest(store, symbols, sm, skip_broken=universe != "sp500")
    closes = close_panel(bars)
    sp500_members = membership_from_additions(sp500_addition_dates(CONSTITUENTS), closes.index)
    members = closes.notna().cummax()
    for col in sp500_members.columns.intersection(members.columns):
        members[col] = sp500_members[col]
    cons = pd.concat([pd.read_csv(p) for p in ALL_CONSTITUENTS if Path(p).exists()])
    cons = cons.drop_duplicates("Symbol")
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
    high = pd.DataFrame({k: v["high"] for k, v in bars.items()}).reindex(closes.index)
    low = pd.DataFrame({k: v["low"] for k, v in bars.items()}).reindex(closes.index)
    spread = corwin_schultz_spread(high, low)
    if features in ("price", "all", "extended"):
        factors.update(price_factor_frames(closes))
    if features in ("fundamental", "all", "extended"):
        fund = fundamental_factor_frames(
            items_for(load_facts(), present), split, panel(Adjustment.RAW)
        )
        factors.update({k: fund[k] for k in FUNDAMENTAL_FACTORS if k in fund})
    if features == "extended":
        earn = earnings_factor_frames(items_for(load_facts(), present), closes.index)
        factors.update({k: earn[k] for k in ("earnings_surprise", "surprise_freshness")})
        volumes = pd.DataFrame({k: v["volume"] for k, v in bars.items()}).reindex(closes.index)
        factors.update(volume_factor_frames(closes, volumes))
    return ResearchData(
        bars,
        sm,
        closes,
        split,
        members,
        store.read("SPY")["close"],
        sectors,
        factors,
        spread,
    )
