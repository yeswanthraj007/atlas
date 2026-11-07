"""Market intraday momentum (pre-registered: research/preregistered/2026-11-07-intraday-
momentum-v1.md). Everything is built from 1-minute bars; a bar labelled HH:MM covers
HH:MM:00 to HH:MM:59, so "the price at 10:00" is the CLOSE of the 09:59 bar.

Decision time is 15:30. The signal uses only bars up to the 15:29 bar's close; the trade
enters at the OPEN of the 15:30 bar and exits at the CLOSE of the 15:59 bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"
# minutes since midnight of the bars used
B_0959, B_1459, B_1529, B_1530, B_1559 = 599, 899, 929, 930, 959


def session_table(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per complete regular session: previous close, prices at the decision points,
    and the returns the strategies need. Days missing any needed bar are dropped."""
    et = bars.index.tz_convert(NY)
    df = bars.assign(date=et.normalize().tz_localize(None), tod=et.hour * 60 + et.minute)
    df = df[(df["tod"] >= 570) & (df["tod"] <= B_1559)]

    def at(tod: int, col: str) -> pd.Series:
        return df[df["tod"] == tod].set_index("date")[col]

    t = pd.DataFrame(
        {
            "close_1559": at(B_1559, "close"),
            "p1000": at(B_0959, "close"),
            "p1500": at(B_1459, "close"),
            "p1530": at(B_1529, "close"),
            "entry": at(B_1530, "open"),
        }
    ).dropna()
    t["prev_close"] = t["close_1559"].shift(1)
    # only use a previous close from the immediately preceding session in the table
    t["r1"] = np.log(t["p1000"] / t["prev_close"])
    t["r12"] = np.log(t["p1530"] / t["p1500"])
    t["r13"] = np.log(t["close_1559"] / t["p1530"])  # the paper's last half-hour return
    t["trade_ret"] = np.log(t["close_1559"] / t["entry"])  # what the strategy earns
    daily = np.log(t["close_1559"] / t["close_1559"].shift(1))
    t["vol20"] = daily.rolling(20, min_periods=15).std().shift(1)  # known at the start of the day
    return t.dropna(subset=["r1", "r12", "r13", "trade_ret"])  # vol20 may be NaN early on


def daily_pnl(position: pd.Series, trade_ret: pd.Series, cost_bp: float) -> pd.Series:
    """Net return per day: position x trade return minus the round-trip cost when trading."""
    return position * trade_ret - position.abs() * cost_bp / 1e4


def summarize(pnl: pd.Series, position: pd.Series) -> dict:
    """Stats over all days (flat days count as zero) and over traded days."""
    traded = pnl[position != 0]
    n = len(traded)
    mean, sd = traded.mean(), traded.std(ddof=1)
    allsd = pnl.std(ddof=1)
    eq = pnl.cumsum()
    return {
        "days": len(pnl),
        "trades": int(n),
        "mean_net_bp": float(mean * 1e4) if n else float("nan"),
        "t": float(mean / (sd / np.sqrt(n))) if n > 2 and sd > 0 else float("nan"),
        "hit_rate": float((traded > 0).mean()) if n else float("nan"),
        "sharpe": float(pnl.mean() / allsd * np.sqrt(252)) if allsd > 0 else float("nan"),
        "total_net_pct": float(pnl.sum() * 100),
        "max_drawdown_pct": float((eq - eq.cummax()).min() * 100),
    }
