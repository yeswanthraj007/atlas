"""Security master: one canonical definition per instrument, shared by every module.

Instrument ids are stable, human-readable keys ("EQ:AAPL", "OPT:AAPL:2026-12-18:C:200").
Vendor identifiers (FIGI, ISIN, CUSIP, broker symbols) are attributes, never keys, so a
vendor change never rewrites history in the ledger.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from atlas.core.types import AssetClass


class OptionRight(StrEnum):
    CALL = "C"
    PUT = "P"


class Instrument(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    symbol: str
    asset_class: AssetClass
    currency: str = "USD"
    multiplier: Decimal = Decimal(1)
    identifiers: dict[str, str] = Field(default_factory=dict)  # figi, isin, cusip, alpaca, ...
    sector: str | None = None


class Equity(Instrument):
    asset_class: AssetClass = AssetClass.EQUITY

    @classmethod
    def of(cls, symbol: str, **kw: object) -> Equity:
        return cls(id=f"EQ:{symbol}", symbol=symbol, **kw)  # type: ignore[arg-type]


class ETF(Instrument):
    asset_class: AssetClass = AssetClass.ETF

    @classmethod
    def of(cls, symbol: str, **kw: object) -> ETF:
        return cls(id=f"ETF:{symbol}", symbol=symbol, **kw)  # type: ignore[arg-type]


class Option(Instrument):
    asset_class: AssetClass = AssetClass.OPTION
    multiplier: Decimal = Decimal(100)
    underlying_id: str
    strike: Decimal
    expiry: date
    right: OptionRight

    @classmethod
    def of(
        cls, underlying: Instrument, expiry: date, right: OptionRight, strike: Decimal | int
    ) -> Option:
        strike = Decimal(strike)
        occ = f"{underlying.symbol}{expiry:%y%m%d}{right.value}{int(strike * 1000):08d}"
        return cls(
            id=f"OPT:{underlying.symbol}:{expiry.isoformat()}:{right.value}:{strike}",
            symbol=occ,
            underlying_id=underlying.id,
            strike=strike,
            expiry=expiry,
            right=right,
            currency=underlying.currency,
        )


class Future(Instrument):
    asset_class: AssetClass = AssetClass.FUTURE
    underlying_id: str | None = None
    expiry: date


class FXPair(Instrument):
    asset_class: AssetClass = AssetClass.FX
    base: str
    quote: str


class Bond(Instrument):
    asset_class: AssetClass = AssetClass.BOND
    coupon: Decimal  # annual rate, e.g. Decimal("0.045")
    maturity: date
    face: Decimal = Decimal(100)
    frequency: int = 2


class UnknownInstrumentError(KeyError):
    pass


class SecurityMaster:
    """In-memory registry. A Postgres-backed implementation keeps the same interface."""

    def __init__(self, instruments: list[Instrument] | None = None) -> None:
        self._by_id: dict[str, Instrument] = {}
        self._by_symbol: dict[str, str] = {}
        for inst in instruments or []:
            self.add(inst)

    def add(self, inst: Instrument) -> Instrument:
        existing = self._by_id.get(inst.id)
        if existing is not None and existing != inst:
            raise ValueError(f"conflicting definition for {inst.id}")
        self._by_id[inst.id] = inst
        self._by_symbol.setdefault(inst.symbol, inst.id)
        return inst

    def get(self, instrument_id: str) -> Instrument:
        try:
            return self._by_id[instrument_id]
        except KeyError:
            raise UnknownInstrumentError(instrument_id) from None

    def by_symbol(self, symbol: str) -> Instrument:
        return self.get(self._by_symbol[symbol])

    def multiplier(self, instrument_id: str) -> Decimal:
        return self.get(instrument_id).multiplier

    def __contains__(self, instrument_id: object) -> bool:
        return instrument_id in self._by_id

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)
