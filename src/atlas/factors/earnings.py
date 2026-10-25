"""Earnings-event features from SEC filing history (point-in-time).

* earnings_surprise: (TTM net income now - TTM a quarter earlier) / total assets, i.e.
  this quarter's income minus the same quarter last year, scaled. Usable from the filing
  date + 1 business day. Post-earnings drift: Bernard & Thomas (1989).
* surprise_freshness: earnings_surprise while the filing is under 63 trading days old
  (the drift fades over roughly that window), else 0.
* days_since_filing: trading days since the latest 10-Q / 10-K became usable. Lets a model
  learn the quarterly cycle: volatility jumps around earnings, ~60-90 days after the
  previous report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.marketdata.sec import instant_series, point_in_time, ttm_series

FRESH_DAYS = 63


def _filing_dates(facts: pd.DataFrame) -> pd.DatetimeIndex:
    f = facts[facts["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])]
    return pd.DatetimeIndex(sorted(set(pd.to_datetime(f["filed"]))))


def company_earnings_series(facts: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, pd.Series]:
    ttm = ttm_series(facts, "net_income").sort_values("end").drop_duplicates("end")
    nan = pd.Series(np.nan, index=dates)
    if len(ttm) >= 2:
        # consecutive quarter ends ~ 91 days apart: the TTM change is Q(t) - Q(t-4)
        gap = ttm["end"].diff().dt.days
        change = ttm["val"].diff().where(gap.between(80, 100))
        surprise = ttm.assign(val=change, known=ttm["known"]).dropna(subset=["val"])
        surprise_pit = point_in_time(surprise[["end", "val", "known"]], dates)
    else:
        surprise_pit = nan
    assets = point_in_time(instant_series(facts, "assets"), dates)
    es = surprise_pit / assets.where(assets > 0)

    filed = _filing_dates(facts)
    if len(filed):
        usable = filed + pd.offsets.BDay(1)
        if dates.tz is not None:
            usable = usable.tz_localize(dates.tz)
        pos_in_dates = np.searchsorted(dates.values, usable.values)  # first usable trading day
        last = pd.Series(np.nan, index=dates)
        for p in pos_in_dates:
            if p < len(dates):
                last.iloc[p] = p
        last = last.ffill()
        days_since = pd.Series(np.arange(len(dates)), index=dates) - last
    else:
        days_since = nan
    fresh = (days_since <= FRESH_DAYS).astype(float).where(days_since.notna())
    return {
        "earnings_surprise": es,
        "surprise_freshness": es * fresh,
        "days_since_filing": days_since,
    }


def earnings_factor_frames(
    facts_by_symbol: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, prefix: str = "EQ:"
) -> dict[str, pd.DataFrame]:
    per: dict[str, dict[str, pd.Series]] = {}
    for sym, facts in facts_by_symbol.items():
        for name, s in company_earnings_series(facts, dates).items():
            per.setdefault(name, {})[prefix + sym] = s
    return {k: pd.DataFrame(v, index=dates) for k, v in per.items()}


def volume_factor_frames(closes: pd.DataFrame, volumes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Liquidity and attention features (higher = more attractive, by literature sign).

    * abnormal_volume: log(21-day avg volume / 252-day avg volume). High attention.
    * illiquidity: log Amihud (2002) ratio, mean(|return| / dollar volume) over 21 days.
      Illiquid stocks have historically earned a premium.
    """
    v = volumes.reindex_like(closes)
    r = closes.pct_change(fill_method=None)
    dollar = (closes * v).replace(0.0, np.nan)
    amihud = (r.abs() / dollar).rolling(21, min_periods=15).mean()
    return {
        "abnormal_volume": np.log(
            v.rolling(21, min_periods=15).mean() / v.rolling(252, min_periods=200).mean()
        ),
        "illiquidity": np.log(amihud.replace(0.0, np.nan)),
    }
