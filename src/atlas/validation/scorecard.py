"""Where does a strategy work, and where does it fail?

* `ticker_scorecard`  — every round trip (open to flat) per ticker: P&L, win rate, holding time
* `yearly_table`      — calendar-year returns vs a benchmark
* `regime_table`      — performance in bull/bear trends and low/mid/high volatility
* `drawdown_table`    — the worst peak-to-trough episodes and how long they lasted
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal

import numpy as np
import pandas as pd

from atlas.core.types import ZERO
from atlas.ledger.ledger import Transaction, TxnKind

TRADING_DAYS = 252


def round_trips(txns: Iterable[Transaction], final_marks: Mapping[str, Decimal]) -> pd.DataFrame:
    """Split each instrument's trades into round trips: from flat, to flat again.

    A trip still open at the end is closed at `final_marks` (flagged `open_at_end`).
    Return per trip = P&L / gross capital bought into the trip.
    """
    by_inst: dict[str, list[Transaction]] = {}
    for t in txns:
        if t.kind is TxnKind.TRADE and t.instrument_id and t.cancels is None:
            by_inst.setdefault(t.instrument_id, []).append(t)
    rows = []
    for inst, ts in by_inst.items():
        ts.sort(key=lambda t: t.effective_at)
        qty, cash, invested, opened = ZERO, ZERO, ZERO, None
        for t in ts:
            if qty == ZERO:
                opened, cash, invested = t.effective_at, ZERO, ZERO
            qty += t.quantity
            cash -= t.quantity * t.price + t.fee
            if t.quantity > 0:
                invested += t.quantity * t.price
            if qty == ZERO:
                rows.append((inst, opened, t.effective_at, cash, invested, False))
        if qty != ZERO and inst in final_marks:
            pnl = cash + qty * final_marks[inst]
            rows.append((inst, opened, ts[-1].effective_at, pnl, invested, True))
    df = pd.DataFrame(
        rows, columns=["instrument", "opened", "closed", "pnl", "invested", "open_at_end"]
    )
    if df.empty:
        return df
    df["pnl"] = df["pnl"].astype(float)
    df["invested"] = df["invested"].astype(float)
    df["return"] = np.where(df["invested"] > 0, df["pnl"] / df["invested"], np.nan)
    df["days_held"] = (pd.to_datetime(df["closed"]) - pd.to_datetime(df["opened"])).dt.days
    return df


def ticker_scorecard(trips: pd.DataFrame) -> pd.DataFrame:
    if trips.empty:
        return pd.DataFrame()
    g = trips.groupby("instrument")
    out = pd.DataFrame(
        {
            "trips": g.size(),
            "total_pnl": g["pnl"].sum(),
            "win_rate": g["pnl"].apply(lambda x: float((x > 0).mean())),
            "avg_return": g["return"].mean(),
            "best_return": g["return"].max(),
            "worst_return": g["return"].min(),
            "avg_days_held": g["days_held"].mean(),
        }
    )
    total = out["total_pnl"].sum()
    out["pnl_share"] = out["total_pnl"] / total if total else np.nan
    return out.sort_values("total_pnl", ascending=False)


def yearly_table(strategy: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Calendar-year total returns from equity curves."""

    def by_year(eq: pd.Series) -> pd.Series:
        r = eq.pct_change().dropna()
        return (1 + r).groupby(r.index.year).prod() - 1

    out = pd.DataFrame({"strategy": by_year(strategy)})
    if benchmark is not None:
        out["benchmark"] = by_year(benchmark)
        out["excess"] = out["strategy"] - out["benchmark"]
    return out


def _stats(r: pd.Series) -> dict[str, float]:
    if len(r) < 2:
        return {"days": len(r), "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann = float(r.mean() * TRADING_DAYS)
    return {"days": len(r), "ann_return": ann, "ann_vol": vol, "sharpe": ann / vol if vol else 0}


def regime_table(
    strategy_returns: pd.Series, benchmark_close: pd.Series, trend_window: int = 200
) -> pd.DataFrame:
    """Strategy statistics split by market regime, using only information known that day.

    Trend: benchmark above (bull) or below (bear) its 200-day average at the previous close.
    Volatility: tercile of the benchmark's 21-day realised volatility at the previous close.
    """
    bench = benchmark_close.sort_index()
    trend = (bench > bench.rolling(trend_window).mean()).shift(1)
    vol21 = bench.pct_change().rolling(21).std().shift(1)
    terciles = vol21.quantile([1 / 3, 2 / 3])
    vol_reg = pd.cut(
        vol21,
        [-np.inf, terciles.iloc[0], terciles.iloc[1], np.inf],
        labels=["low_vol", "mid_vol", "high_vol"],
    )
    bench_r = bench.pct_change()
    r = strategy_returns.dropna()
    rows = {}
    for name, mask in [
        ("bull_trend", trend.reindex(r.index) == True),  # noqa: E712 (NaN-safe compare)
        ("bear_trend", trend.reindex(r.index) == False),  # noqa: E712
        *[(lbl, vol_reg.reindex(r.index) == lbl) for lbl in ["low_vol", "mid_vol", "high_vol"]],
    ]:
        mask = mask.fillna(False).astype(bool)
        row = _stats(r[mask])
        row["benchmark_ann_return"] = float(bench_r.reindex(r.index)[mask].mean() * TRADING_DAYS)
        rows[name] = row
    return pd.DataFrame(rows).T


def drawdown_table(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    """Worst drawdown episodes: peak date, trough date, recovery date (if any), depth."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    episodes = []
    in_dd, start = False, None
    for ts, val in dd.items():
        if val < 0 and not in_dd:
            in_dd, start = True, ts
        elif val == 0 and in_dd:
            seg = dd.loc[start:ts]
            episodes.append((start, seg.idxmin(), ts, float(seg.min())))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), None, float(seg.min())))
    df = pd.DataFrame(episodes, columns=["start", "trough", "recovered", "depth"])
    if df.empty:
        return df
    end = pd.to_datetime(df["recovered"]).fillna(equity.index[-1])
    df["days_to_recover"] = (end - pd.to_datetime(df["start"])).dt.days
    return df.sort_values("depth").head(top).reset_index(drop=True)
