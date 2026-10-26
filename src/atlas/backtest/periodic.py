"""Fast periodic portfolio simulation with partial trading and per-stock trading costs.

Garleanu & Pedersen (2013): with trading costs, move only part of the way from the current
portfolio toward the target ("aim") each period:  w_t = w_{t-1} + tau * (aim_t - w_{t-1}).
tau = 1 rebuilds the portfolio every period; smaller tau trades less and pays less.

Weights are set at the close of each rebalance date and earn the return to the next one.
Positions drift with returns between rebalances. Costs: |trade| x half the stock's
round-trip cost (each trade pays one side), charged when the trade happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PeriodicResult:
    gross: pd.Series  # per-period return before costs
    cost: pd.Series  # per-period trading cost
    turnover: pd.Series  # sum of |trades| per period

    @property
    def net(self) -> pd.Series:
        return self.gross - self.cost


def simulate(
    aims: pd.DataFrame, fwd_returns: pd.DataFrame, round_trip_cost: pd.DataFrame, tau: float = 1.0
) -> PeriodicResult:
    """`aims`, `fwd_returns`, `round_trip_cost`: rebalance dates x instruments. `fwd_returns`
    on date d is the return from d to the next rebalance date."""
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    cols = aims.columns
    w = pd.Series(0.0, index=cols)
    gross, cost, turnover = {}, {}, {}
    for d in aims.index:
        aim = aims.loc[d].fillna(0.0)
        target = w + tau * (aim - w)
        trade = (target - w).abs()
        c = round_trip_cost.loc[d].reindex(cols).fillna(round_trip_cost.loc[d].max()) / 2.0
        cost[d] = float((trade * c).sum())
        turnover[d] = float(trade.sum())
        r = fwd_returns.loc[d].reindex(cols).fillna(0.0)
        gross[d] = float((target * r).sum())
        w = target * (1.0 + r)  # drift with returns until the next rebalance
        gross_exposure = target.abs().sum()
        if gross_exposure > 0 and w.abs().sum() > 0:  # keep exposure stable between periods
            w = w * (gross_exposure / w.abs().sum())
    return PeriodicResult(pd.Series(gross), pd.Series(cost), pd.Series(turnover))


def decile_aims(
    pred: pd.DataFrame, long_short: bool = True, top_n: int | None = None
) -> pd.DataFrame:
    """Equal-weight top decile (and short bottom decile), or long-only top_n."""
    rows = {}
    for d, s in pred.iterrows():
        s = s.dropna()
        if len(s) < 20:
            continue
        w = pd.Series(0.0, index=pred.columns)
        if top_n is not None:
            top = s.nlargest(top_n).index
            w[top] = 1.0 / len(top)
        else:
            q = pd.qcut(s.rank(method="first"), 10, labels=False)
            top, bottom = s.index[q == 9], s.index[q == 0]
            w[top] = 1.0 / len(top)
            if long_short:
                w[bottom] = -1.0 / len(bottom)
        rows[d] = w
    return pd.DataFrame(rows).T
