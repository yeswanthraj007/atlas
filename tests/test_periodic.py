import numpy as np
import pandas as pd
import pytest

from atlas.backtest.periodic import decile_aims, simulate

D = pd.bdate_range("2024-01-31", periods=4, freq="21B")
COLS = ["A", "B"]


def frame(v):
    return pd.DataFrame(v, index=D, columns=COLS, dtype=float)


def test_full_rebalance_costs_and_returns() -> None:
    aims = frame([[1, 0], [0, 1], [1, 0], [0, 1]])  # flip every period
    fwd = frame([[0.10, 0.0], [0.0, 0.05], [0.0, 0.0], [0.0, 0.0]])
    cost = frame(np.full((4, 2), 0.002))  # 20 bps round trip -> 10 bps per side
    res = simulate(aims, fwd, cost, tau=1.0)
    assert res.gross.iloc[0] == pytest.approx(0.10)
    assert res.cost.iloc[0] == pytest.approx(0.001)  # buy 100% of A
    assert res.turnover.iloc[1] == pytest.approx(2.0)  # sell A, buy B
    assert res.cost.iloc[1] == pytest.approx(0.002)
    assert res.gross.iloc[1] == pytest.approx(0.05)


def test_partial_trading_cuts_turnover() -> None:
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2020-01-31", periods=60, freq="21B")
    aims = pd.DataFrame(rng.choice([0.0, 0.1], size=(60, 10)), index=d)
    fwd = pd.DataFrame(rng.normal(0, 0.05, (60, 10)), index=d)
    cost = pd.DataFrame(0.001, index=d, columns=aims.columns)
    full = simulate(aims, fwd, cost, tau=1.0)
    half = simulate(aims, fwd, cost, tau=0.5)
    assert half.turnover.sum() < 0.7 * full.turnover.sum()
    assert half.cost.sum() < full.cost.sum()
    with pytest.raises(ValueError):
        simulate(aims, fwd, cost, tau=0)


def test_decile_aims() -> None:
    pred = pd.DataFrame([np.arange(100.0)], index=D[:1], columns=[f"S{i}" for i in range(100)])
    ls = decile_aims(pred)
    assert ls.iloc[0].sum() == pytest.approx(0.0) and ls.iloc[0].abs().sum() == pytest.approx(2)
    assert ls.iloc[0]["S99"] > 0 and ls.iloc[0]["S0"] < 0
    lo = decile_aims(pred, top_n=50)
    assert lo.iloc[0].sum() == pytest.approx(1.0) and (lo.iloc[0] >= 0).all()
