"""Point-in-time fundamental factors from SEC filings. HIGHER = MORE ATTRACTIVE.

| factor              | definition                              | reference                   |
| ------------------- | --------------------------------------- | --------------------------- |
| earnings_yield      | TTM net income / market cap             | Basu 1977                   |
| book_to_price       | equity / market cap (equity > 0)        | Fama & French 1992          |
| sales_to_price      | TTM revenue / market cap                | Barbee et al. 1996          |
| cf_yield            | TTM operating cash flow / market cap    | Lakonishok et al. 1994      |
| gross_profitability | TTM gross profit / total assets         | Novy-Marx 2013              |
| roe                 | TTM net income / equity (equity > 0)    | Fama & French 2015 (RMW)    |
| low_accruals        | -(TTM net income - TTM CFO) / assets    | Sloan 1996                  |
| low_asset_growth    | -(assets / assets a year earlier - 1)   | Cooper, Gulen & Schill 2008 |
| low_leverage        | -(liabilities / assets)                 | quality / distress          |
| revenue_growth      | TTM revenue / TTM revenue a year ago - 1| growth                      |

Market cap on date t = (shares at their report date, converted to today's share units
via that date's raw/split-adjusted price ratio) x split-adjusted close on t. A stock split
between the share-count report and t therefore cannot distort any ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.marketdata.sec import instant_series, point_in_time, ttm_series

YEAR = 252


def _split_factor_at(raw: pd.Series, split_adj: pd.Series, when: pd.Series) -> np.ndarray:
    """raw / split-adjusted close on (or just before) each date in `when`."""
    ratio = (raw / split_adj).dropna()
    if ratio.empty:
        return np.full(len(when), np.nan)
    idx = ratio.index
    w = pd.DatetimeIndex(when)
    if idx.tz is not None and w.tz is None:
        w = w.tz_localize(idx.tz)
    pos = idx.searchsorted(w, side="right") - 1
    out = np.where(pos >= 0, ratio.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return out


def company_factor_series(
    facts: pd.DataFrame,
    split_close: pd.Series,
    raw_close: pd.Series,
    dates: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    ttm = {
        k: point_in_time(ttm_series(facts, k), dates)
        for k in ("net_income", "revenue", "gross_profit", "cfo")
    }
    inst = {
        k: point_in_time(instant_series(facts, k), dates)
        for k in ("assets", "liabilities", "equity")
    }
    sh = instant_series(facts, "shares")
    if not sh.empty:
        sh = sh.assign(
            val=sh["val"].to_numpy() * _split_factor_at(raw_close, split_close, sh["end"])
        )
    shares_today_units = point_in_time(sh.dropna(), dates)
    mcap = shares_today_units * split_close.reindex(dates)
    mcap = mcap.where(mcap > 0)
    equity_pos = inst["equity"].where(inst["equity"] > 0)
    assets = inst["assets"].where(inst["assets"] > 0)
    return {
        "earnings_yield": ttm["net_income"] / mcap,
        "book_to_price": equity_pos / mcap,
        "sales_to_price": ttm["revenue"] / mcap,
        "cf_yield": ttm["cfo"] / mcap,
        "gross_profitability": ttm["gross_profit"] / assets,
        "roe": ttm["net_income"] / equity_pos,
        "low_accruals": -(ttm["net_income"] - ttm["cfo"]) / assets,
        "low_asset_growth": -(assets / assets.shift(YEAR) - 1.0),
        "low_leverage": -(inst["liabilities"] / assets),
        "revenue_growth": ttm["revenue"] / ttm["revenue"].shift(YEAR) - 1.0,
        "market_cap": mcap,
    }


def fundamental_factor_frames(
    facts_by_symbol: dict[str, pd.DataFrame],
    split_closes: pd.DataFrame,
    raw_closes: pd.DataFrame,
    prefix: str = "EQ:",
) -> dict[str, pd.DataFrame]:
    """Factor name -> DataFrame (dates x instrument ids), aligned to `split_closes`."""
    dates = split_closes.index
    per_factor: dict[str, dict[str, pd.Series]] = {}
    for sym, facts in facts_by_symbol.items():
        inst = prefix + sym
        if inst not in split_closes.columns or inst not in raw_closes.columns:
            continue
        series = company_factor_series(facts, split_closes[inst], raw_closes[inst], dates)
        for name, s in series.items():
            per_factor.setdefault(name, {})[inst] = s
    return {
        name: pd.DataFrame(cols, index=dates).replace([np.inf, -np.inf], np.nan)
        for name, cols in per_factor.items()
    }


FUNDAMENTAL_FACTORS = [
    "earnings_yield",
    "book_to_price",
    "sales_to_price",
    "cf_yield",
    "gross_profitability",
    "roe",
    "low_accruals",
    "low_asset_growth",
    "low_leverage",
    "revenue_growth",
]
