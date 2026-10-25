"""Statistics that correct performance numbers for luck.

* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): probability that the true
  Sharpe exceeds a benchmark, given the track-record length and the skew/fat tails of
  the returns.
* Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the PSR against the Sharpe you
  would expect from the *best of N* useless strategies. If you tried 50 settings and kept
  the best, its Sharpe must clear that bar before it means anything.

All Sharpe ratios here are per period (e.g. daily), not annualised.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe(returns: NDArray[np.float64]) -> float:
    r = np.asarray(returns, dtype=np.float64)
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd)


def probabilistic_sharpe(returns: NDArray[np.float64], benchmark_sr: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    if n < 3:
        raise ValueError("need at least 3 returns")
    sr = sharpe(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # non-excess: normal = 3
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe among `n_trials` strategies with zero true skill."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(
    returns: NDArray[np.float64], n_trials: int, trial_sharpes: NDArray[np.float64] | None = None
) -> float:
    """Probability the strategy's skill is real after accounting for `n_trials` attempts.

    `trial_sharpes`: per-period Sharpe ratios of all the variants tried; their variance
    sets how high luck alone could reach. Without them, the variance of the Sharpe
    estimator itself (1 / n) is used, which is the minimum, i.e. a lenient bar.
    """
    r = np.asarray(returns, dtype=np.float64)
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        var = float(np.var(trial_sharpes, ddof=1))
    else:
        var = 1.0 / r.size
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, var))


def alpha_beta(
    returns: NDArray[np.float64], benchmark: NDArray[np.float64], periods_per_year: int = 252
) -> dict[str, float]:
    """CAPM regression r = alpha + beta * b + e. Beating a benchmark with beta > 1 in a
    rising market is leverage, not skill; only alpha (and its t-stat) measures skill."""
    y = np.asarray(returns, dtype=np.float64)
    x = np.asarray(benchmark, dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    sigma2 = float(resid @ resid) / max(len(y) - 2, 1)
    se = np.sqrt(sigma2 * np.diag(np.linalg.inv(design.T @ design)))
    return {
        "alpha_ann": float(coef[0] * periods_per_year),
        "alpha_t": float(coef[0] / se[0]) if se[0] else 0.0,
        "beta": float(coef[1]),
        "residual_vol_ann": float(np.sqrt(sigma2 * periods_per_year)),
    }
