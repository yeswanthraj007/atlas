"""Broker interface and the simulated broker used by backtests and dry runs.

The simulated broker is deliberately pessimistic: orders submitted during bar t can only
fill from bar t+1 (no look-ahead), market orders pay slippage, limit orders fill only when
the bar trades through the limit, and fills are capped at a share of bar volume.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Protocol

from atlas.core.types import ZERO, Side, to_decimal
from atlas.oms.orders import Fill, Order, OrderType, TimeInForce

FillHandler = Callable[[Fill], None]
CancelHandler = Callable[[str, str], None]  # (order_id, reason)


class Broker(Protocol):
    def set_handlers(self, on_fill: FillHandler, on_cancel: CancelHandler) -> None: ...
    def submit(self, order: Order) -> str: ...
    def cancel(self, order_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


@dataclass
class CostModel:
    per_share: Decimal = Decimal("0.005")
    min_per_order: Decimal = Decimal("1.00")
    bps_of_notional: Decimal = ZERO
    slippage_bps: Decimal = Decimal("5")

    def fee(self, qty: Decimal, price: Decimal, multiplier: Decimal) -> Decimal:
        notional = qty * price * multiplier
        return max(
            self.min_per_order, self.per_share * qty
        ) + notional * self.bps_of_notional / Decimal(10_000)


class SimulatedBroker:
    def __init__(
        self,
        multiplier: Callable[[str], Decimal],
        costs: CostModel | None = None,
        max_participation: float | None = 0.10,
        lot_size: Decimal = Decimal(1),
    ) -> None:
        self._multiplier = multiplier
        self.costs = costs or CostModel()
        self.max_participation = max_participation
        self.lot_size = lot_size
        self._working: dict[str, Order] = {}
        self._on_fill: FillHandler = lambda f: None
        self._on_cancel: CancelHandler = lambda oid, r: None
        self._seq = 0

    def set_handlers(self, on_fill: FillHandler, on_cancel: CancelHandler) -> None:
        self._on_fill, self._on_cancel = on_fill, on_cancel

    def submit(self, order: Order) -> str:
        self._seq += 1
        self._working[order.order_id] = order
        return f"sim-{self._seq}"

    def cancel(self, order_id: str) -> None:
        if self._working.pop(order_id, None) is not None:
            self._on_cancel(order_id, "canceled by request")

    @property
    def working_orders(self) -> list[Order]:
        return list(self._working.values())

    def process_bar(self, timestamp: datetime, bars: Mapping[str, Bar]) -> None:
        for order in list(self._working.values()):
            bar = bars.get(order.instrument_id)
            if bar is not None:
                self._try_fill(order, bar, timestamp)
            if order.status.is_terminal:
                self._working.pop(order.order_id, None)
            elif order.request.tif in (TimeInForce.DAY, TimeInForce.IOC) and bar is not None:
                self._working.pop(order.order_id, None)
                self._on_cancel(order.order_id, f"{order.request.tif} order expired unfilled")

    def _try_fill(self, order: Order, bar: Bar, ts: datetime) -> None:
        req = order.request
        slip = float(self.costs.slippage_bps) / 10_000
        if req.order_type is OrderType.MARKET:
            raw = bar.open * (1 + slip) if req.side is Side.BUY else bar.open * (1 - slip)
        else:
            assert req.limit_price is not None
            limit = float(req.limit_price)
            if req.side is Side.BUY:
                if bar.low > limit:
                    return
                raw = min(bar.open, limit)
            else:
                if bar.high < limit:
                    return
                raw = max(bar.open, limit)

        qty = order.remaining
        if self.max_participation is not None and bar.volume is not None:
            cap = Decimal(str(bar.volume * self.max_participation))
            qty = min(qty, (cap / self.lot_size).to_integral_value(ROUND_DOWN) * self.lot_size)
        if qty <= ZERO:
            return
        price = to_decimal(raw)
        fee = self.costs.fee(qty, price, self._multiplier(order.instrument_id))
        self._on_fill(Fill(order.order_id, qty, price, ts, fee=fee.quantize(Decimal("0.01"))))
