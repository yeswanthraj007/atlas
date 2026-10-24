"""Strategy interface.

A strategy sees market history and the current book, and returns either explicit order
requests or target portfolio weights. It never talks to a broker: the OMS (with its
risk checks) sits between every strategy and the market, in backtests and in live.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Protocol

import pandas as pd

from atlas.core.types import ZERO, Side
from atlas.ledger.ledger import PositionBook
from atlas.oms.orders import OrderRequest

TargetWeights = dict[str, float]


@dataclass(frozen=True)
class StrategyContext:
    timestamp: datetime
    portfolio_id: str
    prices: Mapping[str, Decimal]  # latest close per instrument
    history: pd.DataFrame  # closes up to and including `timestamp`, one column per instrument
    book: PositionBook
    nav: Decimal
    multiplier: Callable[[str], Decimal]
    bar_index: int


class Strategy(Protocol):
    id: str

    def on_bar(self, ctx: StrategyContext) -> list[OrderRequest] | TargetWeights | None: ...


def rebalance_orders(
    targets: TargetWeights,
    ctx: StrategyContext,
    strategy_id: str,
    min_trade_value: Decimal = Decimal(100),
    lot_size: Decimal = Decimal(1),
) -> list[OrderRequest]:
    """Turn target weights into the orders that move the book there.

    Instruments held but absent from `targets` are sold to zero. Quantities round toward
    zero to whole lots, so the book never overshoots its targets or its cash. Sells are
    listed first so they free cash before buys fill.
    """
    held = set(ctx.book.positions)
    orders: list[OrderRequest] = []
    for inst in sorted(held | set(targets)):
        px = ctx.prices.get(inst)
        if px is None or px <= ZERO:
            continue
        mult = ctx.multiplier(inst)
        target_qty = Decimal(str(targets.get(inst, 0.0))) * ctx.nav / (px * mult)
        delta = target_qty - ctx.book.quantity(inst)
        delta = (delta / lot_size).to_integral_value(ROUND_DOWN) * lot_size
        if delta == ZERO or abs(delta) * px * mult < min_trade_value:
            continue
        orders.append(
            OrderRequest(
                ctx.portfolio_id,
                inst,
                Side.BUY if delta > 0 else Side.SELL,
                abs(delta),
                strategy_id=strategy_id,
            )
        )
    orders.sort(key=lambda o: o.side is Side.BUY)
    return orders
