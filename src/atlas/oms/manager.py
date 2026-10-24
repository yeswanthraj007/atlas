"""Order manager: the only path from an order request to a broker and into the ledger.

request -> pre-trade risk -> broker -> fills -> ledger -> live position book
Every step is written to the audit log.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from atlas.core.types import ZERO, Side, utcnow
from atlas.execution.broker import Broker
from atlas.ledger.ledger import Ledger, PositionBook, Transaction, TxnKind
from atlas.oms.orders import Fill, Order, OrderRequest, OrderStatus, OrderType
from atlas.oms.risk_checks import PreTradeContext, PreTradeRiskEngine
from atlas.refdata.instruments import SecurityMaster


@dataclass(frozen=True, slots=True)
class AuditEvent:
    at: datetime
    event: str
    order_id: str | None
    detail: dict[str, Any] = field(default_factory=dict)


class OrderManager:
    def __init__(
        self,
        portfolio_id: str,
        ledger: Ledger,
        secmaster: SecurityMaster,
        broker: Broker,
        risk: PreTradeRiskEngine,
        prices: Callable[[], Mapping[str, Decimal]],
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.portfolio_id = portfolio_id
        self.ledger = ledger
        self.secmaster = secmaster
        self.broker = broker
        self.risk = risk
        self._prices = prices
        self._clock = clock
        self.orders: dict[str, Order] = {}
        self.audit: list[AuditEvent] = []
        self._day_start_nav: Decimal | None = None
        self._book = PositionBook.build(
            ledger.transactions(portfolio_id), portfolio_id, secmaster.multiplier
        )
        ledger.subscribe(self._on_txn)
        broker.set_handlers(self.on_fill, self.on_broker_cancel)

    # ---- state -----------------------------------------------------------------
    @property
    def book(self) -> PositionBook:
        return self._book

    @property
    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if not o.status.is_terminal]

    def nav(self) -> Decimal:
        return self._book.nav(self._prices(), self.secmaster.multiplier)

    def start_day(self) -> None:
        """Mark the NAV that today's P&L (and the daily loss limit) is measured from."""
        self._day_start_nav = self.nav()
        self._log("start_day", None, nav=str(self._day_start_nav))

    def day_pnl(self) -> Decimal:
        return ZERO if self._day_start_nav is None else self.nav() - self._day_start_nav

    def _on_txn(self, txn: Transaction) -> None:
        if txn.portfolio_id != self.portfolio_id:
            return
        if txn.cancels is not None:  # corrections change history: rebuild from the ledger
            self._book = PositionBook.build(
                self.ledger.transactions(self.portfolio_id),
                self.portfolio_id,
                self.secmaster.multiplier,
            )
        else:
            self._book._apply(txn, self.secmaster.multiplier)
            self._book.positions = {
                k: p for k, p in self._book.positions.items() if p.quantity != ZERO
            }

    def context(self) -> PreTradeContext:
        prices = self._prices()
        return PreTradeContext(
            book=self._book,
            prices=prices,
            nav=self._book.nav(prices, self.secmaster.multiplier),
            multiplier=self.secmaster.multiplier,
            open_orders=self.open_orders,
            day_pnl=self.day_pnl(),
            now=self._clock(),
        )

    # ---- order flow ------------------------------------------------------------
    def submit(self, request: OrderRequest) -> Order:
        if request.portfolio_id != self.portfolio_id:
            raise ValueError("order for a different portfolio")
        self.secmaster.get(request.instrument_id)  # unknown instruments never reach a broker
        now = self._clock()
        order = Order(request, created_at=now, updated_at=now)
        # Evaluate before registering, so the order is not counted among working orders.
        violations = self.risk.evaluate(order, self.context())
        self.orders[order.order_id] = order
        if violations:
            order.reject_reasons = violations
            order.transition(OrderStatus.REJECTED, now)
            self._log("rejected", order.order_id, reasons=violations, **self._describe(request))
            return order
        order.transition(OrderStatus.NEW, now)
        self._log("accepted", order.order_id, **self._describe(request))
        order.broker_order_id = self.broker.submit(order)
        return order

    def cancel(self, order_id: str) -> None:
        order = self.orders[order_id]
        if not order.status.is_terminal:
            self.broker.cancel(order_id)

    def on_fill(self, fill: Fill) -> None:
        order = self.orders[fill.order_id]
        order.apply_fill(fill)
        self.ledger.append(
            Transaction(
                portfolio_id=self.portfolio_id,
                kind=TxnKind.TRADE,
                instrument_id=order.instrument_id,
                quantity=fill.quantity * order.request.side.sign,
                price=fill.price,
                fee=fill.fee,
                effective_at=fill.timestamp,
                recorded_at=fill.timestamp,
                order_id=order.order_id,
                currency=self.secmaster.get(order.instrument_id).currency,
            )
        )
        self._log(
            "fill",
            order.order_id,
            qty=str(fill.quantity),
            price=str(fill.price),
            fee=str(fill.fee),
            status=order.status.value,
        )

    def on_broker_cancel(self, order_id: str, reason: str) -> None:
        order = self.orders.get(order_id)
        if order is None or order.status.is_terminal:
            return
        expired = "expired" in reason
        order.transition(OrderStatus.EXPIRED if expired else OrderStatus.CANCELED, self._clock())
        self._log("expired" if expired else "canceled", order_id, reason=reason)

    def flatten_all(self, strategy_id: str = "flatten") -> list[Order]:
        """Market orders closing every position. Allowed even with the kill switch on."""
        for o in self.open_orders:
            self.cancel(o.order_id)
        out = []
        for inst, pos in list(self._book.positions.items()):
            side = Side.SELL if pos.quantity > 0 else Side.BUY
            out.append(
                self.submit(
                    OrderRequest(
                        self.portfolio_id,
                        inst,
                        side,
                        abs(pos.quantity),
                        OrderType.MARKET,
                        strategy_id=strategy_id,
                    )
                )
            )
        return out

    # ---- helpers ---------------------------------------------------------------
    def _describe(self, r: OrderRequest) -> dict[str, Any]:
        return {
            "instrument": r.instrument_id,
            "side": r.side.value,
            "qty": str(r.quantity),
            "type": r.order_type.value,
            "limit": str(r.limit_price) if r.limit_price else None,
            "strategy": r.strategy_id,
        }

    def _log(self, event: str, order_id: str | None, **detail: Any) -> None:
        self.audit.append(AuditEvent(self._clock(), event, order_id, detail))
