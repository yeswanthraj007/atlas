import numpy as np
import pandas as pd
import pytest
from scipy import stats

from atlas.risk import backtesting as bt
from atlas.risk.covariance import ewma_covariance, ledoit_wolf, nearest_psd, sample_covariance
from atlas.risk.factors import fit_factor_model
from atlas.risk.stress import Scenario, run_stress
from atlas.risk.var import (
    component_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
    var_es_from_pnl,
)

RNG = np.random.default_rng(42)
COV = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])  # daily: 2% and 3% vol
EXPO = np.array([1_000_000.0, 500_000.0])


def test_empirical_var_es_on_known_distribution() -> None:
    pnl = -np.arange(1, 101, dtype=float)  # losses 1..100
    var, es = var_es_from_pnl(pnl, 0.95)
    assert var == 96.0
    assert es == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_parametric_matches_closed_form() -> None:
    est = parametric_var(EXPO, COV, 0.99)
    sigma = np.sqrt(EXPO @ COV @ EXPO)
    assert est.var == pytest.approx(stats.norm.ppf(0.99) * sigma)
    assert est.es > est.var
    ten_day = parametric_var(EXPO, COV, 0.99, horizon_days=10)
    assert ten_day.var == pytest.approx(est.var * np.sqrt(10))


def test_monte_carlo_normal_converges_to_parametric() -> None:
    mc = monte_carlo_var(EXPO, COV, 0.99, n_sims=400_000, dof=None, seed=1)
    pa = parametric_var(EXPO, COV, 0.99)
    assert mc.var == pytest.approx(pa.var, rel=0.02)
    assert mc.es == pytest.approx(pa.es, rel=0.03)


def test_student_t_has_fatter_tail_at_same_variance() -> None:
    t = monte_carlo_var(EXPO, COV, 0.999, n_sims=400_000, dof=4, seed=2)
    n = monte_carlo_var(EXPO, COV, 0.999, n_sims=400_000, dof=None, seed=2)
    assert t.var > n.var


def test_historical_var_on_normal_sample() -> None:
    rets = RNG.multivariate_normal([0, 0], COV, size=200_000)
    h = historical_var(EXPO, rets, 0.99)
    assert h.var == pytest.approx(parametric_var(EXPO, COV, 0.99).var, rel=0.03)
    h10 = historical_var(EXPO, rets[:5000], 0.99, horizon_days=10)
    assert h10.n_scenarios == 4991


def test_component_var_sums_to_total() -> None:
    comp = component_var(EXPO, COV, 0.99)
    assert comp.sum() == pytest.approx(parametric_var(EXPO, COV, 0.99).var)


def test_covariance_estimators() -> None:
    rets = RNG.multivariate_normal([0, 0], COV, size=100_000)
    assert sample_covariance(rets) == pytest.approx(COV, rel=0.1, abs=1e-5)
    # EWMA reacts to a recent volatility spike; the sample estimate barely moves
    spiked = np.vstack([rets[:2000], rets[:20] * 4.0])
    assert ewma_covariance(spiked, 0.94)[0, 0] > 3 * sample_covariance(spiked)[0, 0]
    lw, s = ledoit_wolf(rets)
    assert 0.0 <= s <= 1.0
    assert lw == pytest.approx(COV, rel=0.15, abs=2e-5)
    # Many assets, few observations: shrinkage kicks in and fixes conditioning
    wide = RNG.standard_normal((30, 50)) * 0.01
    lw_wide, s_wide = ledoit_wolf(wide)
    assert s_wide > 0.2
    assert np.linalg.eigvalsh(lw_wide).min() > 0
    assert np.linalg.eigvalsh(nearest_psd(np.array([[1, 1.2], [1.2, 1.0]]))).min() > 0


def test_kupiec_and_christoffersen() -> None:
    n = 1000
    good = np.zeros(n, dtype=bool)
    good[np.linspace(0, n - 1, 10, dtype=int)] = True  # 10 exceptions, spread out
    assert bt.kupiec_pof(good, 0.99).passed
    assert bt.christoffersen_independence(good).passed
    too_many = np.zeros(n, dtype=bool)
    too_many[:40] = True  # 40 exceptions, all clustered
    assert not bt.kupiec_pof(too_many, 0.99).passed
    assert not bt.christoffersen_independence(too_many).passed
    assert not bt.conditional_coverage(too_many, 0.99).passed
    zero = np.zeros(n, dtype=bool)
    assert np.isfinite(bt.kupiec_pof(zero, 0.99).statistic)


def test_basel_traffic_light() -> None:
    hits = np.zeros(250, dtype=bool)
    assert bt.basel_traffic_light(hits) is bt.TrafficLight.GREEN
    hits[:6] = True
    assert bt.basel_traffic_light(hits) is bt.TrafficLight.YELLOW
    hits[:12] = True
    assert bt.basel_traffic_light(hits) is bt.TrafficLight.RED


def test_factor_model_recovers_betas() -> None:
    idx = pd.date_range("2020-01-01", periods=2000, freq="B")
    mkt = pd.Series(RNG.normal(0, 0.01, len(idx)), idx, name="MKT")
    true = {"A": 1.5, "B": 0.5}
    assets = pd.DataFrame(
        {k: b * mkt + RNG.normal(0, 0.005, len(idx)) for k, b in true.items()}, index=idx
    )
    fm = fit_factor_model(assets, mkt.to_frame())
    assert fm.betas.loc["A", "MKT"] == pytest.approx(1.5, abs=0.05)
    assert fm.betas.loc["B", "MKT"] == pytest.approx(0.5, abs=0.05)
    decomp = fm.risk_decomposition(pd.Series({"A": 1e6, "B": 1e6}))
    assert 0.0 < decomp["factor_share"] < 1.0
    assert decomp["total_vol"] ** 2 == pytest.approx(
        decomp["factor_vol"] ** 2 + decomp["specific_vol"] ** 2
    )


def test_stress_factor_and_instrument_shocks() -> None:
    expo = pd.Series({"A": 1_000_000.0, "B": -500_000.0, "C": 200_000.0})
    betas = pd.DataFrame({"MKT": [1.2, 0.8, 0.0]}, index=["A", "B", "C"])
    s = Scenario("crash", instrument_shocks={"C": -0.5}, factor_shocks={"MKT": -0.2})
    res = run_stress(expo, s, betas)
    assert res.pnl_by_position["A"] == pytest.approx(-240_000)
    assert res.pnl_by_position["B"] == pytest.approx(80_000)
    assert res.pnl_by_position["C"] == pytest.approx(-100_000)
    with pytest.raises(ValueError):
        run_stress(expo, s)
