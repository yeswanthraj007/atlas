"""Value-at-Risk and Expected Shortfall.

Sign convention: VaR and ES are reported as positive numbers meaning *loss*, in the
same currency as the exposures. `confidence=0.99` means the 99% one-period VaR.

Inputs are dollar exposures (quantity x price x multiplier for linear instruments, or
delta-equivalent exposure for options) and asset returns. Non-linear instruments get
full revaluation through the `revalue` hook of `monte_carlo_var`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class RiskEstimate:
    method: str
    confidence: float
    horizon_days: int
    var: float
    es: float
    n_scenarios: int


def _check_confidence(confidence: float) -> None:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")


def var_es_from_pnl(pnl: Vector, confidence: float) -> tuple[float, float]:
    """Empirical VaR/ES from a vector of scenario P&Ls (gains positive)."""
    _check_confidence(confidence)
    losses = -np.asarray(pnl, dtype=np.float64)
    if losses.size == 0:
        raise ValueError("no scenarios")
    var = float(np.quantile(losses, confidence, method="higher"))
    tail = losses[losses >= var]
    return var, float(tail.mean())


def historical_var(
    exposures: Vector, returns: Matrix, confidence: float = 0.99, horizon_days: int = 1
) -> RiskEstimate:
    """Historical simulation: replay each past return vector on today's exposures.

    For horizon > 1 we use overlapping h-day compounded returns instead of sqrt-time
    scaling, so fat tails and autocorrelation in the history are preserved.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    if horizon_days > 1:
        growth = np.log1p(r)
        csum = np.vstack([np.zeros(r.shape[1]), np.cumsum(growth, axis=0)])
        r = np.expm1(csum[horizon_days:] - csum[:-horizon_days])
    pnl = r @ np.asarray(exposures, dtype=np.float64)
    var, es = var_es_from_pnl(pnl, confidence)
    return RiskEstimate("historical", confidence, horizon_days, var, es, len(pnl))


def parametric_var(
    exposures: Vector,
    cov: Matrix,
    confidence: float = 0.99,
    horizon_days: int = 1,
    mean: Vector | None = None,
) -> RiskEstimate:
    """Delta-normal VaR and the closed-form normal ES."""
    _check_confidence(confidence)
    w = np.asarray(exposures, dtype=np.float64)
    sigma = float(np.sqrt(w @ cov @ w * horizon_days))
    mu = 0.0 if mean is None else float(w @ mean) * horizon_days
    z = stats.norm.ppf(confidence)
    var = z * sigma - mu
    es = sigma * stats.norm.pdf(z) / (1.0 - confidence) - mu
    return RiskEstimate("parametric", confidence, horizon_days, var, es, 0)


def monte_carlo_var(
    exposures: Vector,
    cov: Matrix,
    confidence: float = 0.99,
    horizon_days: int = 1,
    n_sims: int = 100_000,
    dof: float | None = 5.0,
    seed: int | None = 0,
    revalue: Callable[[Matrix], Vector] | None = None,
) -> RiskEstimate:
    """Simulated returns from a multivariate normal (dof=None) or Student-t (fat tails).

    `revalue(sim_returns) -> pnl` replaces the linear P&L for books with options: price
    every position under every simulated move (full revaluation).
    """
    rng = np.random.default_rng(seed)
    n = cov.shape[0]
    chol = np.linalg.cholesky(cov * horizon_days + 1e-18 * np.eye(n))
    z = rng.standard_normal((n_sims, n))
    if dof is not None:
        if dof <= 2:
            raise ValueError("Student-t dof must be > 2 for finite variance")
        # Scale so the simulated covariance still equals `cov`.
        chi = rng.chisquare(dof, size=(n_sims, 1))
        z = z * np.sqrt((dof - 2.0) / chi)
    sims = z @ chol.T
    pnl = revalue(sims) if revalue is not None else sims @ np.asarray(exposures)
    var, es = var_es_from_pnl(pnl, confidence)
    return RiskEstimate("monte_carlo", confidence, horizon_days, var, es, n_sims)


def component_var(exposures: Vector, cov: Matrix, confidence: float = 0.99) -> Vector:
    """Euler allocation of parametric VaR. Components sum exactly to total VaR, so the
    dashboard can say which positions the risk actually comes from."""
    w = np.asarray(exposures, dtype=np.float64)
    sigma = float(np.sqrt(w @ cov @ w))
    if sigma == 0.0:
        return np.zeros_like(w)
    return stats.norm.ppf(confidence) * w * (cov @ w) / sigma


def marginal_var(exposures: Vector, cov: Matrix, confidence: float = 0.99) -> Vector:
    """dVaR/dExposure: VaR added per extra dollar in each position (pre-trade what-if)."""
    w = np.asarray(exposures, dtype=np.float64)
    sigma = float(np.sqrt(w @ cov @ w))
    if sigma == 0.0:
        return np.zeros_like(w)
    return stats.norm.ppf(confidence) * (cov @ w) / sigma
