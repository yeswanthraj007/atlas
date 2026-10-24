"""Covariance estimators.

* `sample_covariance` — unbiased sample estimate.
* `ewma_covariance`  — RiskMetrics (J.P. Morgan 1996): zero-mean, exponentially weighted,
                       lambda = 0.94 for daily data. Reacts quickly to volatility regimes.
* `ledoit_wolf`      — Ledoit & Wolf (2004) shrinkage toward a scaled identity. Use when
                       the number of assets is large relative to the history, where the
                       sample matrix is ill-conditioned and optimizers exploit its noise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]


def _as_2d(returns: NDArray[np.float64]) -> Matrix:
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("returns must be a T x N array with T >= 2")
    if not np.isfinite(x).all():
        raise ValueError("returns contain NaN or inf; clean or align them first")
    return x


def sample_covariance(returns: NDArray[np.float64]) -> Matrix:
    return np.atleast_2d(np.cov(_as_2d(returns), rowvar=False, ddof=1))


def ewma_covariance(returns: NDArray[np.float64], lam: float = 0.94) -> Matrix:
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")
    x = _as_2d(returns)
    t = x.shape[0]
    weights = (1.0 - lam) * lam ** np.arange(t - 1, -1, -1, dtype=np.float64)
    weights /= weights.sum()  # finite-sample normalisation
    return (x * weights[:, None]).T @ x


def ledoit_wolf(returns: NDArray[np.float64]) -> tuple[Matrix, float]:
    """Returns (shrunk covariance, shrinkage intensity in [0, 1])."""
    x = _as_2d(returns)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    x2 = x**2
    trace_s = float(np.sum(x2) / t)
    mu = trace_s / n
    delta_ = float(np.sum((x.T @ x) ** 2)) / t**2
    beta_ = float(np.sum(x2.T @ x2))
    beta = (beta_ / t - delta_) / (n * t)
    delta = (delta_ - 2.0 * mu * trace_s + n * mu**2) / n
    beta = min(beta, delta)
    shrinkage = 0.0 if delta == 0 else float(max(beta, 0.0) / delta)
    return (1.0 - shrinkage) * s + shrinkage * mu * np.eye(n), shrinkage


def correlation_from_covariance(cov: Matrix) -> Matrix:
    vol = np.sqrt(np.diag(cov))
    return cov / np.outer(vol, vol)


def nearest_psd(cov: Matrix, eps: float = 1e-12) -> Matrix:
    """Clip negative eigenvalues — needed after stress-shocking a correlation matrix."""
    sym = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    return (vecs * np.maximum(vals, eps)) @ vecs.T
