"""Shared primitive types.

Accounting quantities (quantities, prices, cash) are `Decimal` so the ledger never
accumulates float error. Risk and analytics work in float64 numpy arrays and convert
at the boundary with `to_decimal` / `float(...)`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from uuid import uuid4

ZERO = Decimal(0)
PRICE_QUANTUM = Decimal("0.000001")


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"
    FX = "fx"
    BOND = "bond"
    CASH = "cash"
    CRYPTO = "crypto"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


def to_decimal(value: float | int | str | Decimal, quantum: Decimal = PRICE_QUANTUM) -> Decimal:
    """Convert to Decimal via str() so 0.1 becomes Decimal('0.1'), not its binary expansion."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_EVEN)


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"
