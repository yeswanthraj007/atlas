"""Pre-trade risk and compliance checks.

Every order — from a human, a strategy or the AI layer — passes through the same rules
before it can reach a broker. Strategies cannot see or skip this layer.

A rule returns None when the order passes, or a human-readable reason when it fails.
Rules that cap *risk* exempt orders that only *reduce* an existing position, so the kill
switch or a daily-loss breach never traps you in a position you are trying to exit.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol

from atlas.core.types import ZERO
from atlas.ledger.ledger import PositionBook
from atlas.oms.orders import Order, OrderType


@dataclass
class PreTradeContext:
    book: PositionBook
    prices: Mapping[str, Decimal]
    nav: Decimal
    multiplier: Callable[[str], Decimal]
    open_orders: list[Order]
    day_pnl: Decimal
    now: datetime

    def pending_quantity(self, instrument_id: str) -> Decimal:
        return sum(
            (
                o.remaining * o.request.side.sign
                for o in self.open_orders
                if o.instrument_id == instrument_id
            ),
            ZERO,
        )

    def post_trade_quantity(self, order: Order) -> Decimal:
        return (
            self.book.quantity(order.instrument_id)
            + self.pending_quantity(order.instrument_id)
            + order.request.signed_quantity
        )

    def reference_price(self, order: Order) -> Decimal:
        px = self.prices.get(order.instrument_id)
        if px is None:
            if order.request.limit_price is not None:
                return order.request.limit_price
            raise KeyError(f"no reference price for {order.instrument_id}")
        return px

    def is_reducing(self, order: Order) -> bool:
        """True if the order only shrinks the absolute position (never flips it)."""
        current = self.book.quantity(order.instrument_id) + self.pending_quantity(
            order.instrument_id
        )
        after = current + order.request.signed_quantity
        return (
            current != ZERO
            and abs(after) < abs(current)
            and (after == ZERO or (after > 0) == (current > 0))
        )


class Rule(Protocol):
    name: str

    def check(self, order: Order, ctx: PreTradeContext) -> str | None: ...


@dataclass
class KillSwitch:
    """Global stop. When engaged, only position-reducing orders are accepted."""

    name: str = "kill_switch"
    engaged: bool = False
    reason: str = ""

    def engage(self, reason: str) -> None:
        self.engaged, self.reason = True, reason

    def release(self) -> None:
        self.engaged, self.reason = False, ""

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        if self.engaged and not ctx.is_reducing(order):
            return f"kill switch engaged: {self.reason}"
        return None


@dataclass
class RestrictedList:
    instruments: set[str]
    name: str = "restricted_list"

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        if order.instrument_id in self.instruments and not ctx.is_reducing(order):
            return f"{order.instrument_id} is on the restricted list"
        return None


@dataclass
class MaxOrderNotional:
    limit: Decimal
    name: str = "max_order_notional"

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        notional = (
            order.request.quantity
            * ctx.reference_price(order)
            * ctx.multiplier(order.instrument_id)
        )
        if notional > self.limit:
            return f"order notional {notional:,.0f} exceeds {self.limit:,.0f}"
        return None


@dataclass
class MaxPositionWeight:
    """Post-trade |position| as a fraction of NAV, including other working orders."""

    limit: Decimal
    name: str = "max_position_weight"

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        if ctx.is_reducing(order):
            return None
        qty = ctx.post_trade_quantity(order)
        value = abs(qty) * ctx.reference_price(order) * ctx.multiplier(order.instrument_id)
        if ctx.nav <= ZERO:
            return "NAV is not positive"
        weight = value / ctx.nav
        if weight > self.limit:
            return (
                f"post-trade weight {weight:.2%} in {order.instrument_id} exceeds {self.limit:.2%}"
            )
        return None


@dataclass
class MaxGrossExposure:
    """Post-trade gross exposure (sum of |market value|) as a multiple of NAV."""

    limit: Decimal
    name: str = "max_gross_exposure"

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        if ctx.is_reducing(order):
            return None
        quantities = {i: p.quantity for i, p in ctx.book.positions.items()}
        for o in ctx.open_orders:
            quantities[o.instrument_id] = (
                quantities.get(o.instrument_id, ZERO) + o.remaining * o.request.side.sign
            )
        quantities[order.instrument_id] = (
            quantities.get(order.instrument_id, ZERO) + order.request.signed_quantity
        )
        gross = ZERO
        for inst, qty in quantities.items():
            px = ctx.prices.get(inst)
            if px is None:
                if inst == order.instrument_id:
                    px = ctx.reference_price(order)
                else:
                    return f"no price for {inst}; cannot compute gross exposure"
            gross += abs(qty) * px * ctx.multiplier(inst)
        if ctx.nav <= ZERO:
            return "NAV is not positive"
        leverage = gross / ctx.nav
        if leverage > self.limit:
            return f"post-trade gross exposure {leverage:.2f}x NAV exceeds {self.limit}x"
        return None


@dataclass
class MaxDailyLoss:
    """Once today's loss reaches the limit, only risk-reducing orders are accepted."""

    limit: Decimal  # positive currency amount
    name: str = "max_daily_loss"

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        if ctx.day_pnl <= -self.limit and not ctx.is_reducing(order):
            return f"daily loss {-ctx.day_pnl:,.0f} has hit the {self.limit:,.0f} limit"
        return None


@dataclass
class PriceCollar:
    """Fat-finger guard: a limit price too far from the last trade is almost always a typo."""

    max_deviation: Decimal  # e.g. Decimal("0.05") = 5%
    name: str = "price_collar"

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        if order.request.order_type is not OrderType.LIMIT:
            return None
        last = ctx.prices.get(order.instrument_id)
        if last is None or last == ZERO:
            return None
        assert order.request.limit_price is not None
        dev = abs(order.request.limit_price - last) / last
        if dev > self.max_deviation:
            return f"limit {order.request.limit_price} is {dev:.1%} from last {last}"
        return None


@dataclass
class NoShortSelling:
    name: str = "no_short_selling"

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        if ctx.post_trade_quantity(order) < ZERO:
            return f"order would leave a short position in {order.instrument_id}"
        return None


@dataclass
class OrderRateLimit:
    """Runaway-algorithm guard: at most `max_orders` accepted per `window`."""

    max_orders: int
    window: timedelta = timedelta(minutes=1)
    name: str = "order_rate_limit"
    _times: deque[datetime] = field(default_factory=deque)

    def check(self, order: Order, ctx: PreTradeContext) -> str | None:
        while self._times and ctx.now - self._times[0] > self.window:
            self._times.popleft()
        if len(self._times) >= self.max_orders:
            return f"more than {self.max_orders} orders in {self.window}"
        return None

    def record(self, at: datetime) -> None:
        self._times.append(at)


class PreTradeRiskEngine:
    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules

    def evaluate(self, order: Order, ctx: PreTradeContext) -> list[str]:
        violations: list[str] = []
        for rule in self.rules:
            try:
                reason = rule.check(order, ctx)
            except Exception as exc:  # a broken rule must fail closed, never open
                reason = f"{rule.name} errored: {exc}"
            if reason:
                violations.append(f"[{rule.name}] {reason}")
        if not violations:
            for rule in self.rules:
                if isinstance(rule, OrderRateLimit):
                    rule.record(ctx.now)
        return violations
