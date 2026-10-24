from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest

from atlas.ledger.ledger import Ledger, PositionBook, Transaction, TxnKind
from atlas.refdata.instruments import Equity, Option, OptionRight, SecurityMaster

T0 = datetime(2026, 1, 2, 15, tzinfo=UTC)
PID = "p1"
AAPL = Equity.of("AAPL")
SM = SecurityMaster([AAPL])


def trade(qty: str, px: str, day: int, fee: str = "0", recorded: int | None = None) -> Transaction:
    return Transaction(
        PID,
        TxnKind.TRADE,
        D(qty),
        T0 + timedelta(days=day),
        AAPL.id,
        D(px),
        D(fee),
        recorded_at=T0 + timedelta(days=recorded if recorded is not None else day),
    )


def book(ledger: Ledger, **kw) -> PositionBook:
    return PositionBook.build(ledger.transactions(), PID, SM.multiplier, **kw)


def test_average_cost_and_realized_pnl() -> None:
    lg = Ledger()
    lg.append(Transaction(PID, TxnKind.CASH, D(10_000), T0, recorded_at=T0))
    lg.append(trade("10", "100", 1, fee="1"))
    lg.append(trade("10", "110", 2, fee="1"))
    lg.append(trade("-5", "120", 3, fee="1"))
    b = book(lg)
    pos = b.positions[AAPL.id]
    assert pos.quantity == D(15)
    assert pos.avg_cost == D(105)
    assert b.realized_pnl == D(75)  # 5 * (120 - 105)
    assert b.cash["USD"] == D(10_000) - D(1000) - D(1100) + D(600) - D(3)
    assert b.fees == D(3)
    # NAV identity: cash + market value
    assert b.nav({AAPL.id: D(120)}, SM.multiplier) == b.cash["USD"] + D(15) * D(120)


def test_flip_through_zero_resets_cost() -> None:
    lg = Ledger()
    lg.append(trade("10", "100", 1))
    lg.append(trade("-15", "90", 2))
    pos = book(lg).positions[AAPL.id]
    assert pos.quantity == D(-5)
    assert pos.avg_cost == D(90)
    assert pos.realized_pnl == D(-100)


def test_closed_position_disappears() -> None:
    lg = Ledger()
    lg.append(trade("10", "100", 1))
    lg.append(trade("-10", "100", 2))
    assert AAPL.id not in book(lg).positions


def test_bitemporal_as_of_and_known_at() -> None:
    lg = Ledger()
    lg.append(trade("10", "100", day=1))
    lg.append(trade("5", "100", day=2, recorded=5))  # booked late
    assert book(lg, as_of=T0 + timedelta(days=1)).quantity(AAPL.id) == D(10)
    assert book(lg, as_of=T0 + timedelta(days=3)).quantity(AAPL.id) == D(15)
    # what we believed on day 3 about day 3: the late trade wasn't known yet
    as_known = book(lg, as_of=T0 + timedelta(days=3), known_at=T0 + timedelta(days=3))
    assert as_known.quantity(AAPL.id) == D(10)


def test_cancel_reverses_without_deleting() -> None:
    lg = Ledger()
    bad = lg.append(trade("100", "100", 1))
    lg.cancel(bad.txn_id, recorded_at=T0 + timedelta(days=2))
    assert len(lg) == 2
    assert book(lg).quantity(AAPL.id) == D(0)
    assert book(lg, known_at=T0 + timedelta(days=1)).quantity(AAPL.id) == D(100)


def test_ledger_rejects_bad_input() -> None:
    lg = Ledger()
    t = lg.append(trade("1", "1", 1))
    with pytest.raises(ValueError):
        lg.append(t)  # duplicate id
    with pytest.raises(ValueError):
        Transaction(PID, TxnKind.TRADE, D(0), T0, AAPL.id, D(1))
    with pytest.raises(ValueError):
        Transaction(PID, TxnKind.TRADE, D(1), T0, None, D(1))


def test_option_multiplier_in_cash_and_pnl() -> None:
    from datetime import date

    opt = Option.of(AAPL, date(2026, 12, 18), OptionRight.CALL, 200)
    sm = SecurityMaster([AAPL, opt])
    assert opt.symbol == "AAPL261218C00200000"
    lg = Ledger()
    lg.append(Transaction(PID, TxnKind.TRADE, D(2), T0, opt.id, D("5.00")))
    lg.append(Transaction(PID, TxnKind.TRADE, D(-2), T0 + timedelta(days=1), opt.id, D("7.50")))
    b = PositionBook.build(lg.transactions(), PID, sm.multiplier)
    assert b.realized_pnl == D(500)  # 2 contracts * 2.50 * 100
    assert b.cash["USD"] == D(500)
