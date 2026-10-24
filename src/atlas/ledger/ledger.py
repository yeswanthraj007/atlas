"""Append-only transaction ledger and the position book (IBOR) derived from it.

Rules that make the book auditable:
  * Transactions are never edited or deleted. A mistake is fixed by appending a
    transaction whose `cancels` field names the bad one.
  * Every transaction is bitemporal: `effective_at` is when it happened in the market,
    `recorded_at` is when we learned of it. `PositionBook.build(as_of, known_at)` answers
    both "what did we hold on date X" and "what did we *think* we held on date X, as
    known at time Y" — the question every reconciliation and audit eventually asks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from atlas.core.types import ZERO, new_id, utcnow


class TxnKind(StrEnum):
    TRADE = "trade"  # quantity signed (+buy / -sell), price per unit
    CASH = "cash"  # deposit (+) / withdrawal (-); quantity is the cash amount
    FEE = "fee"  # standalone fee; quantity is the (positive) amount charged
    DIVIDEND = "dividend"  # cash income on instrument_id; quantity is the cash amount
    TRANSFER = "transfer"  # position moved in/out at a stated cost price


@dataclass(frozen=True, slots=True)
class Transaction:
    portfolio_id: str
    kind: TxnKind
    quantity: Decimal
    effective_at: datetime
    instrument_id: str | None = None
    price: Decimal = ZERO
    fee: Decimal = ZERO
    currency: str = "USD"
    recorded_at: datetime = field(default_factory=utcnow)
    order_id: str | None = None
    cancels: str | None = None
    note: str = ""
    txn_id: str = field(default_factory=lambda: new_id("txn"))

    def __post_init__(self) -> None:
        if self.kind in (TxnKind.TRADE, TxnKind.DIVIDEND, TxnKind.TRANSFER) and not (
            self.instrument_id
        ):
            raise ValueError(f"{self.kind} transaction needs an instrument_id")
        if self.kind is TxnKind.TRADE and self.quantity == ZERO:
            raise ValueError("trade quantity must be non-zero")
        if self.price < ZERO or self.fee < ZERO:
            raise ValueError("price and fee must be non-negative")


class Ledger:
    def __init__(self) -> None:
        self._txns: list[Transaction] = []
        self._ids: set[str] = set()
        self._listeners: list[Callable[[Transaction], None]] = []

    def append(self, txn: Transaction) -> Transaction:
        if txn.txn_id in self._ids:
            raise ValueError(f"duplicate txn_id {txn.txn_id}")
        if txn.cancels is not None and txn.cancels not in self._ids:
            raise ValueError(f"cannot cancel unknown txn {txn.cancels}")
        self._txns.append(txn)
        self._ids.add(txn.txn_id)
        for listener in self._listeners:
            listener(txn)
        return txn

    def cancel(self, txn_id: str, recorded_at: datetime | None = None, note: str = "") -> None:
        original = next(t for t in self._txns if t.txn_id == txn_id)
        self.append(
            Transaction(
                portfolio_id=original.portfolio_id,
                kind=original.kind,
                quantity=original.quantity,
                effective_at=original.effective_at,
                instrument_id=original.instrument_id,
                currency=original.currency,
                recorded_at=recorded_at or utcnow(),
                cancels=txn_id,
                note=note or f"cancel {txn_id}",
            )
        )

    def subscribe(self, listener: Callable[[Transaction], None]) -> None:
        self._listeners.append(listener)

    def transactions(self, portfolio_id: str | None = None) -> list[Transaction]:
        if portfolio_id is None:
            return list(self._txns)
        return [t for t in self._txns if t.portfolio_id == portfolio_id]

    def __len__(self) -> int:
        return len(self._txns)


@dataclass(slots=True)
class Position:
    instrument_id: str
    quantity: Decimal = ZERO
    avg_cost: Decimal = ZERO  # per unit, excluding multiplier
    realized_pnl: Decimal = ZERO

    def apply_trade(self, qty: Decimal, price: Decimal, multiplier: Decimal) -> Decimal:
        """Average-cost accounting. Returns the realized P&L of this trade."""
        realized = ZERO
        if self.quantity == ZERO or (self.quantity > 0) == (qty > 0):
            new_qty = self.quantity + qty
            self.avg_cost = (self.quantity * self.avg_cost + qty * price) / new_qty
            self.quantity = new_qty
        else:
            closing = min(abs(qty), abs(self.quantity))
            direction = Decimal(1) if self.quantity > 0 else Decimal(-1)
            realized = closing * (price - self.avg_cost) * direction * multiplier
            self.quantity += qty
            if self.quantity == ZERO:
                self.avg_cost = ZERO
            elif (self.quantity > 0) != (direction > 0):  # flipped through zero
                self.avg_cost = price
        self.realized_pnl += realized
        return realized


@dataclass(slots=True)
class PositionBook:
    portfolio_id: str
    as_of: datetime | None
    positions: dict[str, Position] = field(default_factory=dict)
    cash: dict[str, Decimal] = field(default_factory=dict)
    realized_pnl: Decimal = ZERO
    income: Decimal = ZERO
    fees: Decimal = ZERO

    @classmethod
    def build(
        cls,
        txns: Iterable[Transaction],
        portfolio_id: str,
        multiplier: Callable[[str], Decimal],
        as_of: datetime | None = None,
        known_at: datetime | None = None,
    ) -> PositionBook:
        visible = [
            t
            for t in txns
            if t.portfolio_id == portfolio_id
            and (as_of is None or t.effective_at <= as_of)
            and (known_at is None or t.recorded_at <= known_at)
        ]
        cancelled = {t.cancels for t in visible if t.cancels}
        live = [t for t in visible if t.cancels is None and t.txn_id not in cancelled]
        live.sort(key=lambda t: (t.effective_at, t.recorded_at))

        book = cls(portfolio_id=portfolio_id, as_of=as_of)
        for t in live:
            book._apply(t, multiplier)
        book.positions = {k: p for k, p in book.positions.items() if p.quantity != ZERO}
        return book

    def _apply(self, t: Transaction, multiplier: Callable[[str], Decimal]) -> None:
        ccy = t.currency
        self.cash.setdefault(ccy, ZERO)
        if t.kind is TxnKind.TRADE:
            assert t.instrument_id is not None
            mult = multiplier(t.instrument_id)
            pos = self.positions.setdefault(t.instrument_id, Position(t.instrument_id))
            self.realized_pnl += pos.apply_trade(t.quantity, t.price, mult)
            self.cash[ccy] -= t.quantity * t.price * mult + t.fee
            self.fees += t.fee
        elif t.kind is TxnKind.TRANSFER:
            assert t.instrument_id is not None
            pos = self.positions.setdefault(t.instrument_id, Position(t.instrument_id))
            pos.apply_trade(t.quantity, t.price, multiplier(t.instrument_id))
        elif t.kind is TxnKind.CASH:
            self.cash[ccy] += t.quantity
        elif t.kind is TxnKind.FEE:
            self.cash[ccy] -= t.quantity
            self.fees += t.quantity
        elif t.kind is TxnKind.DIVIDEND:
            self.cash[ccy] += t.quantity
            self.income += t.quantity

    def quantity(self, instrument_id: str) -> Decimal:
        pos = self.positions.get(instrument_id)
        return pos.quantity if pos else ZERO

    def market_values(
        self, prices: Mapping[str, Decimal], multiplier: Callable[[str], Decimal]
    ) -> dict[str, Decimal]:
        missing = [i for i in self.positions if i not in prices]
        if missing:
            raise KeyError(f"no price for {missing}")
        return {i: p.quantity * prices[i] * multiplier(i) for i, p in self.positions.items()}

    def nav(
        self,
        prices: Mapping[str, Decimal],
        multiplier: Callable[[str], Decimal],
        currency: str = "USD",
    ) -> Decimal:
        # Single-currency for now; FX conversion plugs in here.
        return self.cash.get(currency, ZERO) + sum(
            self.market_values(prices, multiplier).values(), ZERO
        )

    def unrealized_pnl(
        self, prices: Mapping[str, Decimal], multiplier: Callable[[str], Decimal]
    ) -> Decimal:
        return sum(
            (
                p.quantity * (prices[i] - p.avg_cost) * multiplier(i)
                for i, p in self.positions.items()
            ),
            ZERO,
        )
