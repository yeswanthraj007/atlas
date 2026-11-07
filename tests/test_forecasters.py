import numpy as np
import pandas as pd
import torch

from atlas.ml.forecasters import (
    ArmaGarch,
    LSTMBackbone,
    LSTMVol,
    arma_garch_forecast,
    fit_arma_garch,
    qlike,
    score_return_task,
)


def _simulate(n=3000, seed=0, mu=0.03, phi=0.2, theta=0.1, omega=0.05, alpha=0.08, beta=0.9):
    rng = np.random.default_rng(seed)
    sbar = omega / (1 - alpha - beta)
    v, e, r = sbar, 0.0, np.zeros(n)
    prev_e, prev_r = 0.0, mu
    for t in range(n):
        v = omega + alpha * prev_e**2 + beta * v if t else sbar
        e = np.sqrt(v) * rng.normal()
        r[t] = mu + phi * (prev_r - mu) + theta * prev_e + e
        prev_e, prev_r = e, r[t]
    return r


def test_garch_horizon_formula_matches_brute_force_expectation():
    m = ArmaGarch(mu=0.02, phi=0.3, theta=0.1, omega=0.05, alpha=0.08, beta=0.9)
    r = _simulate(500, seed=1)
    p = np.array([400])
    ret, vol = arma_garch_forecast(m, r, p, h=20)
    # brute force: iterate the expected variance and the expected mean day by day
    from scipy.signal import lfilter

    e = lfilter([1, -m.phi], [1, m.theta], r - m.mu)
    v = np.empty(len(r) + 1)
    v[0] = m.long_run_var
    for t in range(1, len(r) + 1):
        v[t] = m.omega + m.alpha * e[t - 1] ** 2 + m.beta * v[t - 1]
    var, tot = v[401], 0.0
    daily_var = []
    for _ in range(20):
        daily_var.append(var)
        var = m.omega + (m.alpha + m.beta) * var  # E[e^2] = var
    mean = m.mu + m.phi * (r[400] - m.mu) + m.theta * e[400]
    for _ in range(20):
        tot += mean
        mean = m.mu + m.phi * (mean - m.mu)
    assert np.isclose(vol[0], np.sqrt(np.mean(daily_var) / 1e4 * 252))
    assert np.isclose(ret[0], tot / 100.0)


def test_forecast_at_p_ignores_the_future():
    m = ArmaGarch(0.02, 0.3, 0.1, 0.05, 0.08, 0.9)
    r = _simulate(600, seed=2)
    r2 = r.copy()
    r2[451:] *= 5.0
    a = arma_garch_forecast(m, r, np.array([300, 450]))
    b = arma_garch_forecast(m, r2, np.array([300, 450]))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y)


def test_fit_recovers_garch_persistence_and_returns_none_for_short_series():
    r = _simulate(4000, seed=3)
    m = fit_arma_garch(r)
    assert m is not None
    assert 0.9 < m.alpha + m.beta < 1.0 and abs(m.beta - 0.9) < 0.1
    assert fit_arma_garch(r[:100]) is None


def test_lstm_backbone_works_with_the_shared_interfaces():
    torch.manual_seed(0)
    b = LSTMBackbone(n_in=10, width=16, layers=1)
    x = torch.randn(4, 120, 10)
    e = torch.zeros(4, 120, 64)
    assert b.hidden(x, e).shape == (4, 120, 16)
    assert LSTMVol(b)(x, e).shape == (4,)
    # causal: changing later days leaves earlier hidden states unchanged
    h1 = b.hidden(x, e)
    x2 = x.clone()
    x2[:, 90:] += 3.0
    assert torch.allclose(h1[:, :90], b.hidden(x2, e)[:, :90], atol=1e-6)


def test_qlike_and_return_scoring():
    t = np.log(np.array([0.2, 0.4]))
    assert np.allclose(qlike(t, t), 0.0) and (qlike(t + 0.2, t) > 0).all()
    dates = pd.to_datetime(["2024-01-01"] * 6 + ["2024-02-01"] * 6)
    fwd = np.tile(np.arange(6.0), 2)
    df = pd.DataFrame({"date": dates, "fwd_ret": fwd})
    s = score_return_task(df, fwd)  # perfect ranking
    assert abs(s["ic"] - 1.0) < 1e-9 and s["accuracy"] == 1.0
