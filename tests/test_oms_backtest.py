from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.core.types import Side
from atlas.execution.broker import Bar, CostModel, SimulatedBroker
from atlas.ledger.ledger import Ledger, Transaction, TxnKind
from atlas.oms.manager import OrderManager
from atlas.oms.orders import (
    Fill,
    InvalidTransitionError,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from atlas.oms.risk_checks import (
    KillSwitch,
    MaxDailyLoss,
    MaxGrossExposure,
    MaxOrderNotional,
    MaxPositionWeight,
    NoShortSelling,
    OrderRateLimit,
    PreTradeRiskEngine,
    PriceCollar,
    RestrictedList,
)
from atlas.refdata.instruments import Equity, SecurityMaster
from atlas.strategy.base import StrategyContext
from atlas.strategy.momentum import CrossSectionalMomentum

T0 = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
AAPL, MSFT = Equity.of("AAPL"), Equity.of("MSFT")
PID = "pm1"
FREE = CostModel(per_share=D(0), min_per_order=D(0), slippage_bps=D(0))


def make_oms(rules, cash="100000", prices=None):
    sm = SecurityMaster([AAPL, MSFT])
    lg = Ledger()
    lg.append(Transaction(PID, TxnKind.CASH, D(cash), T0, recorded_at=T0))
    marks = prices or {AAPL.id: D(100), MSFT.id: D(400)}
    broker = SimulatedBroker(sm.multiplier, FREE, max_participation=None)
    clock = [T0]
    oms = OrderManager(
        PID, lg, sm, broker, PreTradeRiskEngine(rules), lambda: marks, lambda: clock[0]
    )
    return oms, broker, marks, clock


def buy(inst, qty, **kw):
    return OrderRequest(PID, inst.id, Side.BUY, D(qty), **kw)


def sell(inst, qty, **kw):
    return OrderRequest(PID, inst.id, Side.SELL, D(qty), **kw)


def bar(px):
    return Bar(px, px, px, px, 1e9)


def test_order_lifecycle_flows_into_ledger_and_book() -> None:
    oms, broker, *_ = make_oms([])
    o = oms.submit(buy(AAPL, 100))
    assert o.status is OrderStatus.NEW
    broker.process_bar(T0, {AAPL.id: bar(101.0)})
    assert o.status is OrderStatus.FILLED and o.avg_fill_price == D(101)
    assert oms.book.quantity(AAPL.id) == D(100)
    assert oms.book.cash["USD"] == D(100000) - D(10100)
    assert [e.event for e in oms.audit] == ["accepted", "fill"]
    assert oms.ledger.transactions()[-1].order_id == o.order_id


def test_state_machine_and_overfill_guards() -> None:
    o = Order(buy(AAPL, 10))
    with pytest.raises(InvalidTransitionError):
        o.transition(OrderStatus.FILLED)
    o.transition(OrderStatus.NEW)
    o.apply_fill(Fill(o.order_id, D(4), D(10), T0))
    assert o.status is OrderStatus.PARTIALLY_FILLED
    with pytest.raises(ValueError):
        o.apply_fill(Fill(o.order_id, D(7), D(10), T0))
    with pytest.raises(ValueError):
        OrderRequest(PID, AAPL.id, Side.BUY, D(0))
    with pytest.raises(ValueError):
        OrderRequest(PID, AAPL.id, Side.BUY, D(1), OrderType.LIMIT)


@pytest.mark.parametrize(
    ("rule", "request_", "fragment"),
    [
        (MaxOrderNotional(D(5000)), buy(AAPL, 100), "notional"),
        (MaxPositionWeight(D("0.05")), buy(AAPL, 100), "weight"),
        (MaxGrossExposure(D("1.0")), buy(MSFT, 300), "gross"),
        (RestrictedList({AAPL.id}), buy(AAPL, 1), "restricted"),
        (
            PriceCollar(D("0.05")),
            buy(AAPL, 1, order_type=OrderType.LIMIT, limit_price=D(120)),
            "from last",
        ),
        (NoShortSelling(), sell(AAPL, 1), "short"),
    ],
)
def test_each_rule_rejects(rule, request_, fragment) -> None:
    oms, *_ = make_oms([rule])
    o = oms.submit(request_)
    assert o.status is OrderStatus.REJECTED
    assert fragment in o.reject_reasons[0]
    assert oms.audit[-1].event == "rejected"


def test_kill_switch_blocks_new_risk_but_allows_exit() -> None:
    ks = KillSwitch()
    oms, broker, *_ = make_oms([ks])
    oms.submit(buy(AAPL, 50))
    broker.process_bar(T0, {AAPL.id: bar(100.0)})
    ks.engage("manual stop")
    assert oms.submit(buy(AAPL, 1)).status is OrderStatus.REJECTED
    assert oms.submit(sell(AAPL, 80)).status is OrderStatus.REJECTED  # would flip short
    assert oms.submit(sell(AAPL, 20)).status is OrderStatus.NEW  # reduces: allowed
    flat = oms.flatten_all()
    assert all(o.status is OrderStatus.NEW for o in flat)


def test_daily_loss_limit() -> None:
    oms, broker, marks, _ = make_oms([MaxDailyLoss(D(1000))])
    oms.submit(buy(AAPL, 100))
    broker.process_bar(T0, {AAPL.id: bar(100.0)})
    oms.start_day()
    marks[AAPL.id] = D(85)  # -1,500 today
    assert oms.submit(buy(MSFT, 1)).status is OrderStatus.REJECTED
    assert oms.submit(sell(AAPL, 100)).status is OrderStatus.NEW


def test_rate_limit_and_working_orders_count_toward_limits() -> None:
    oms, _, _, clock = make_oms([OrderRateLimit(2, timedelta(minutes=1))])
    assert oms.submit(buy(AAPL, 1)).status is OrderStatus.NEW
    assert oms.submit(buy(AAPL, 1)).status is OrderStatus.NEW
    assert oms.submit(buy(AAPL, 1)).status is OrderStatus.REJECTED
    clock[0] = T0 + timedelta(minutes=2)
    assert oms.submit(buy(AAPL, 1)).status is OrderStatus.NEW

    oms2, *_ = make_oms([MaxPositionWeight(D("0.10"))])
    assert oms2.submit(buy(AAPL, 60)).status is OrderStatus.NEW  # 6% working
    assert oms2.submit(buy(AAPL, 60)).status is OrderStatus.REJECTED  # 12% incl. working


def test_failing_rule_fails_closed() -> None:
    class Broken:
        name = "broken"

        def check(self, order, ctx):
            raise RuntimeError("boom")

    oms, *_ = make_oms([Broken()])
    assert oms.submit(buy(AAPL, 1)).status is OrderStatus.REJECTED


def test_sim_broker_limit_participation_and_expiry() -> None:
    oms, broker, *_ = make_oms([])
    lim = oms.submit(
        buy(AAPL, 10, order_type=OrderType.LIMIT, limit_price=D(95), tif=TimeInForce.GTC)
    )
    broker.process_bar(T0, {AAPL.id: Bar(100, 101, 96, 100)})
    assert lim.status is OrderStatus.NEW  # never traded through 95
    broker.process_bar(T0, {AAPL.id: Bar(97, 98, 94, 95)})
    assert lim.status is OrderStatus.FILLED and lim.avg_fill_price == D(95)

    broker.max_participation = 0.1
    big = oms.submit(buy(AAPL, 500))
    broker.process_bar(T0, {AAPL.id: Bar(100, 100, 100, 100, volume=2000)})
    assert big.filled_quantity == D(200)
    assert big.status is OrderStatus.EXPIRED  # DAY order: remainder expires


def gbm_bars(n_assets=8, n_days=600, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days, tz="UTC")
    out = {}
    sm = SecurityMaster()
    for k in range(n_assets):
        drift = 0.0006 * (k - n_assets / 2)  # some trend up, some down
        r = rng.normal(drift, 0.015, n_days)
        close = 50 * np.exp(np.cumsum(r))
        open_ = close * np.exp(rng.normal(0, 0.003, n_days))
        hi, lo = np.maximum(open_, close) * 1.005, np.minimum(open_, close) * 0.995
        inst = sm.add(Equity.of(f"S{k}"))
        out[inst.id] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": 5e6}, index=idx
        )
    return out, sm


def test_backtest_runs_with_accounting_identity() -> None:
    bars, sm = gbm_bars()
    cfg = BacktestConfig(rules=lambda: [MaxPositionWeight(D("0.30")), MaxGrossExposure(D("1.05"))])
    res = run_backtest(CrossSectionalMomentum(top_n=3), bars, sm, cfg)
    assert res.equity.iloc[0] == pytest.approx(1_000_000)
    assert len(res.ledger) > 10
    final = res.oms.book
    marks = {i: D(str(round(df["close"].iloc[-1], 6))) for i, df in bars.items()}
    nav = final.cash["USD"] + sum((p.quantity * marks[i] for i, p in final.positions.items()), D(0))
    assert float(nav) == pytest.approx(res.equity.iloc[-1], rel=1e-9)
    m = res.metrics()
    assert set(m) >= {"sharpe", "max_drawdown", "cagr"}
    assert m["max_drawdown"] <= 0


def test_backtest_has_no_lookahead() -> None:
    """Orders decided on bar t must fill at bar t+1's open, never at bar t's prices."""
    bars, sm = gbm_bars(n_assets=1, n_days=5)
    inst = next(iter(bars))

    class BuyOnce:
        id = "buy_once"

        def on_bar(self, ctx: StrategyContext):
            if ctx.bar_index == 1:
                return [OrderRequest(ctx.portfolio_id, inst, Side.BUY, D(10))]
            return None

    cfg = BacktestConfig(costs=FREE, max_participation=None)
    res = run_backtest(BuyOnce(), bars, sm, cfg)
    trade = next(t for t in res.ledger.transactions() if t.kind is TxnKind.TRADE)
    assert trade.effective_at == bars[inst].index[2]
    assert float(trade.price) == pytest.approx(bars[inst]["open"].iloc[2], abs=1e-6)
