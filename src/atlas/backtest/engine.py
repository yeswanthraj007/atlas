"""Event-driven backtester that runs strategies through the production OMS.

Per bar t:
  1. start the trading day (daily P&L baseline = previous close NAV)
  2. the simulated broker fills orders submitted at t-1, at bar t's open
  3. marks move to bar t's close
  4. the strategy sees history up to t's close and submits orders (fill at t+1 at best)
  5. NAV is recorded
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from atlas.core.types import to_decimal
from atlas.execution.broker import Bar, CostModel, SimulatedBroker
from atlas.ledger.ledger import Ledger, Transaction, TxnKind
from atlas.oms.manager import OrderManager
from atlas.oms.orders import Order, OrderRequest, OrderStatus
from atlas.oms.risk_checks import PreTradeRiskEngine, Rule
from atlas.refdata.instruments import SecurityMaster
from atlas.strategy.base import Strategy, StrategyContext, rebalance_orders

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    initial_cash: Decimal = Decimal(1_000_000)
    portfolio_id: str = "backtest"
    costs: CostModel = field(default_factory=CostModel)
    max_participation: float | None = 0.10
    rules: Callable[[], list[Rule]] = list  # fresh rule objects per run


@dataclass
class BacktestResult:
    equity: pd.Series
    orders: list[Order]
    ledger: Ledger
    oms: OrderManager

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    @property
    def rejected(self) -> list[Order]:
        return [o for o in self.orders if o.status is OrderStatus.REJECTED]

    def metrics(self) -> dict[str, float]:
        return performance_metrics(self.equity)


def performance_metrics(
    equity: pd.Series, periods_per_year: int = TRADING_DAYS
) -> dict[str, float]:
    r = equity.pct_change().dropna()
    if r.empty:
        return {}
    years = len(r) / periods_per_year
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    downside = float(r[r < 0].std(ddof=1) * np.sqrt(periods_per_year)) if (r < 0).sum() > 1 else 0.0
    mean = float(r.mean() * periods_per_year)
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else -1.0
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": mean / vol if vol else 0.0,
        "sortino": mean / downside if downside else 0.0,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd else 0.0,
    }


def run_backtest(
    strategy: Strategy,
    bars: dict[str, pd.DataFrame],
    secmaster: SecurityMaster,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """`bars`: instrument_id -> DataFrame[open, high, low, close, (volume)] on a DatetimeIndex."""
    cfg = config or BacktestConfig()
    for inst in bars:
        secmaster.get(inst)
    timeline = sorted(set().union(*(df.index for df in bars.values())))
    closes = pd.DataFrame({i: df["close"] for i, df in bars.items()}).sort_index()

    clock: list[datetime] = [timeline[0]]
    marks: dict[str, Decimal] = {}
    ledger = Ledger()
    ledger.append(
        Transaction(
            cfg.portfolio_id,
            TxnKind.CASH,
            cfg.initial_cash,
            timeline[0],
            recorded_at=timeline[0],
            note="initial capital",
        )
    )
    broker = SimulatedBroker(secmaster.multiplier, cfg.costs, cfg.max_participation)
    oms = OrderManager(
        cfg.portfolio_id,
        ledger,
        secmaster,
        broker,
        PreTradeRiskEngine(cfg.rules()),
        prices=lambda: marks,
        clock=lambda: clock[0],
    )

    equity: dict[datetime, float] = {}
    for i, ts in enumerate(timeline):
        clock[0] = ts
        oms.start_day()
        today = {
            inst: Bar(r.open, r.high, r.low, r.close, getattr(r, "volume", None))
            for inst, df in bars.items()
            if ts in df.index
            for r in [df.loc[ts]]
        }
        broker.process_bar(ts, today)
        for inst, bar in today.items():
            marks[inst] = to_decimal(bar.close)

        ctx = StrategyContext(
            timestamp=ts,
            portfolio_id=cfg.portfolio_id,
            prices=dict(marks),
            history=closes.loc[:ts],
            book=oms.book,
            nav=oms.nav(),
            multiplier=secmaster.multiplier,
            bar_index=i,
        )
        decision = strategy.on_bar(ctx)
        requests: list[OrderRequest] = (
            rebalance_orders(decision, ctx, strategy.id)
            if isinstance(decision, dict)
            else list(decision or [])
        )
        for req in requests:
            oms.submit(req)
        equity[ts] = float(oms.nav())

    return BacktestResult(pd.Series(equity, name="nav"), list(oms.orders.values()), ledger, oms)
