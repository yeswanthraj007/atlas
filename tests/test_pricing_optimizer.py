import numpy as np
import pytest

from atlas.optimizer.mean_variance import (
    Constraints,
    OptimizationError,
    mean_variance,
    min_variance,
    risk_contributions,
    risk_parity,
)
from atlas.pricing.black_scholes import bsm, implied_vol


def test_bsm_reference_value_and_parity() -> None:
    c = bsm(100, 100, 1.0, 0.2, r=0.05, is_call=True)
    p = bsm(100, 100, 1.0, 0.2, r=0.05, is_call=False)
    assert float(c.price) == pytest.approx(10.4506, abs=1e-4)  # Hull, textbook value
    assert float(c.price - p.price) == pytest.approx(100 - 100 * np.exp(-0.05), abs=1e-10)


def test_bsm_greeks_match_finite_differences() -> None:
    s, h = 100.0, 1e-3
    g = bsm(s, 105, 0.5, 0.25, r=0.03, q=0.01)
    up, dn = bsm(s + h, 105, 0.5, 0.25, 0.03, 0.01), bsm(s - h, 105, 0.5, 0.25, 0.03, 0.01)
    assert float(g.delta) == pytest.approx(float(up.price - dn.price) / (2 * h), rel=1e-5)
    assert float(g.gamma) == pytest.approx(
        float(up.price - 2 * g.price + dn.price) / h**2, rel=1e-3
    )
    v_up = bsm(s, 105, 0.5, 0.25 + h, 0.03, 0.01)
    assert float(g.vega) == pytest.approx(float(v_up.price - g.price) / h, rel=1e-3)


def test_implied_vol_roundtrip_and_bounds() -> None:
    px = float(bsm(100, 90, 0.75, 0.33, r=0.02, is_call=False).price)
    assert implied_vol(px, 100, 90, 0.75, r=0.02, is_call=False) == pytest.approx(0.33, abs=1e-8)
    with pytest.raises(ValueError):
        implied_vol(200.0, 100, 90, 0.75)


COV = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.01], [0.0, 0.01, 0.16]])
MU = np.array([0.06, 0.08, 0.12])


def test_mean_variance_respects_constraints() -> None:
    res = mean_variance(MU, COV, risk_aversion=2.0, constraints=Constraints(max_weight=0.5))
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (res.weights >= -1e-8).all() and (res.weights <= 0.5 + 1e-6).all()
    capped = mean_variance(MU, COV, 2.0, Constraints(max_vol=0.2))
    assert capped.volatility <= 0.2 + 1e-5


def test_turnover_and_costs_limit_trading() -> None:
    w0 = np.array([1 / 3, 1 / 3, 1 / 3])
    res = mean_variance(MU, COV, 2.0, Constraints(max_turnover=0.1), current=w0)
    assert res.turnover <= 0.1 + 1e-6
    expensive = mean_variance(MU, COV, 2.0, Constraints(tcost=1.0), current=w0)
    assert expensive.turnover == pytest.approx(0.0, abs=1e-5)


def test_infeasible_raises() -> None:
    with pytest.raises(OptimizationError):
        mean_variance(MU, COV, 1.0, Constraints(max_weight=0.2))  # 3 x 0.2 < 1


def test_min_variance_and_risk_parity() -> None:
    mv = min_variance(COV)
    assert mv.weights[0] > mv.weights[2]
    rp = risk_parity(COV)
    assert rp.sum() == pytest.approx(1.0)
    assert risk_contributions(rp, COV) == pytest.approx(np.full(3, 1 / 3), abs=1e-4)
