"""Score a signal on its own, the way an ML model is scored on predictions.

Information coefficient (IC): on each rebalance date, the rank correlation between the
signal and the *next* period's returns across stocks. IC > 0 means the signal ranked
stocks in the right order. A mean IC of 0.02-0.05 with a high hit rate is already useful
in equities; anything much higher deserves suspicion (look-ahead, bad data).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ICSummary:
    n_dates: int
    mean_ic: float
    std_ic: float
    t_stat: float
    hit_rate: float  # share of dates with IC > 0
    ic_ir: float  # mean / std, the signal's "Sharpe"


def signal_panel(
    closes: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series | None],
    rebalance_dates: pd.DatetimeIndex,
    eligible: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate `signal_fn` on history up to each date (never beyond it)."""
    rows = {}
    for ts in rebalance_dates:
        hist = closes.loc[:ts]
        if eligible is not None:
            mask = eligible.loc[:ts].iloc[-1] if len(eligible.loc[:ts]) else None
            if mask is not None:
                hist = hist[[c for c in hist.columns if bool(mask.get(c, False))]]
        sig = signal_fn(hist)
        if sig is not None and len(sig):
            rows[ts] = sig
    return pd.DataFrame(rows).T.sort_index()


def forward_returns(closes: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Return from each date's close to the close `horizon` bars later (NaN at the end)."""
    pos = closes.index.get_indexer(dates)
    out = {}
    for ts, i in zip(dates, pos, strict=True):
        j = i + horizon
        if i < 0 or j >= len(closes):
            continue
        out[ts] = closes.iloc[j] / closes.iloc[i] - 1.0
    return pd.DataFrame(out).T


def information_coefficient(signals: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman IC per date."""
    ics = {}
    for ts in signals.index.intersection(fwd.index):
        s, f = signals.loc[ts], fwd.loc[ts]
        both = pd.concat([s, f], axis=1, keys=["s", "f"]).dropna()
        if len(both) >= 10:
            ics[ts] = stats.spearmanr(both["s"], both["f"]).statistic
    return pd.Series(ics, dtype=float).sort_index()


def summarize_ic(ic: pd.Series) -> ICSummary:
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        raise ValueError("need at least 2 IC observations")
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    return ICSummary(
        n_dates=n,
        mean_ic=mean,
        std_ic=std,
        t_stat=mean / std * np.sqrt(n) if std else 0.0,
        hit_rate=float((ic > 0).mean()),
        ic_ir=mean / std if std else 0.0,
    )


def per_ticker_signal_scores(
    signals: pd.DataFrame, fwd: pd.DataFrame, top_n: int | pd.Series, min_obs: int = 6
) -> pd.DataFrame:
    """How the signal did *for each ticker*.

    * rank_ic: over time, does a higher cross-sectional rank for this stock come before a
      higher relative return? (Spearman of its signal percentile vs return percentile.)
    * times_selected / hit_rate_when_selected: when the stock made the top `top_n`, how
      often it then beat the median stock. `top_n` may vary by date (a Series indexed by
      date), e.g. when walk-forward folds chose different portfolio sizes.
    * avg_excess_when_selected: its average return minus the median, when selected.
    """
    dates = signals.index.intersection(fwd.index)
    sig_pct = signals.loc[dates].rank(axis=1, pct=True)
    fwd_d = fwd.loc[dates]
    fwd_pct = fwd_d.rank(axis=1, pct=True)
    excess = fwd_d.sub(fwd_d.median(axis=1), axis=0)
    limit = top_n.reindex(dates) if isinstance(top_n, pd.Series) else pd.Series(top_n, dates)
    selected = signals.loc[dates].rank(axis=1, ascending=False).le(limit, axis=0)
    rows = []
    for tkr in signals.columns:
        both = pd.concat([sig_pct[tkr], fwd_pct.get(tkr)], axis=1, keys=["s", "f"]).dropna()
        sel = selected[tkr] & excess[tkr].notna() if tkr in excess else selected[tkr] & False
        n_sel = int(sel.sum())
        rows.append(
            {
                "ticker": tkr,
                "obs": len(both),
                "rank_ic": (
                    stats.spearmanr(both["s"], both["f"]).statistic
                    if len(both) >= min_obs and both["s"].nunique() > 1
                    else np.nan
                ),
                "times_selected": n_sel,
                "hit_rate_when_selected": float((excess[tkr][sel] > 0).mean()) if n_sel else np.nan,
                "avg_excess_when_selected": float(excess[tkr][sel].mean()) if n_sel else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
