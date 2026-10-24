"""Order model and lifecycle state machine (mirrors FIX OrdStatus semantics)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from atlas.core.types import ZERO, Side, new_id, utcnow


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


class OrderStatus(StrEnum):
    PENDING_NEW = "pending_new"  # created, not yet risk-checked / acknowledged
    NEW = "new"  # accepted by the broker, working
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
_ALLOWED: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING_NEW: {OrderStatus.NEW, OrderStatus.REJECTED},
    OrderStatus.NEW: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
}


class InvalidTransitionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """What a strategy or a human asks for. Becomes an `Order` once the OMS accepts it."""

    portfolio_id: str
    instrument_id: str
    side: Side
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    tif: TimeInForce = TimeInForce.DAY
    strategy_id: str = "manual"

    def __post_init__(self) -> None:
        if self.quantity <= ZERO:
            raise ValueError("order quantity must be positive; direction comes from side")
        if self.order_type is OrderType.LIMIT and (
            self.limit_price is None or self.limit_price <= ZERO
        ):
            raise ValueError("limit orders need a positive limit_price")

    @property
    def signed_quantity(self) -> Decimal:
        return self.quantity * self.side.sign


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    quantity: Decimal  # positive
    price: Decimal
    timestamp: datetime
    fee: Decimal = ZERO
    fill_id: str = field(default_factory=lambda: new_id("fill"))


@dataclass(slots=True)
class Order:
    request: OrderRequest
    order_id: str = field(default_factory=lambda: new_id("ord"))
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled_quantity: Decimal = ZERO
    avg_fill_price: Decimal = ZERO
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    reject_reasons: list[str] = field(default_factory=list)
    broker_order_id: str | None = None

    @property
    def remaining(self) -> Decimal:
        return self.request.quantity - self.filled_quantity

    @property
    def instrument_id(self) -> str:
        return self.request.instrument_id

    def transition(self, new: OrderStatus, at: datetime | None = None) -> None:
        if new not in _ALLOWED.get(self.status, set()):
            raise InvalidTransitionError(f"{self.order_id}: {self.status} -> {new} not allowed")
        self.status = new
        self.updated_at = at or utcnow()

    def apply_fill(self, fill: Fill) -> None:
        if fill.quantity <= ZERO:
            raise ValueError("fill quantity must be positive")
        if fill.quantity > self.remaining:
            raise ValueError(
                f"{self.order_id}: overfill ({fill.quantity} > remaining {self.remaining})"
            )
        notional = self.avg_fill_price * self.filled_quantity + fill.price * fill.quantity
        self.filled_quantity += fill.quantity
        self.avg_fill_price = notional / self.filled_quantity
        self.transition(
            OrderStatus.FILLED if self.remaining == ZERO else OrderStatus.PARTIALLY_FILLED,
            fill.timestamp,
        )
