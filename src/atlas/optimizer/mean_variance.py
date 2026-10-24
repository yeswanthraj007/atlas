"""Convex portfolio construction (CVXPY).

maximise   mu'w  -  (gamma/2) w'Sigma w  -  tc' |w - w0|
subject to the constraints in `Constraints`.

Weights are fractions of NAV. Transaction costs enter as a linear penalty on trades, so
the optimizer only rebalances when the expected gain beats the cost (Boyd et al. 2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass
class Constraints:
    budget: float | None = 1.0  # sum of weights; None = unconstrained (e.g. long/short overlay)
    long_only: bool = True
    max_weight: float | None = None
    min_weight: float | None = None
    max_gross: float | None = None  # sum |w|, i.e. leverage
    max_vol: float | None = None  # per-period volatility cap on the portfolio
    max_turnover: float | None = None  # sum |w - w0|
    tcost: float | Vector = 0.0  # cost per unit traded, e.g. 0.0005 for 5 bps
    group_caps: dict[str, tuple[list[int], float]] = field(
        default_factory=dict
    )  # name -> (idx, cap)


@dataclass(frozen=True)
class OptResult:
    weights: Vector
    status: str
    expected_return: float
    volatility: float
    turnover: float


class OptimizationError(RuntimeError):
    pass


def mean_variance(
    mu: Vector,
    cov: Matrix,
    risk_aversion: float = 1.0,
    constraints: Constraints | None = None,
    current: Vector | None = None,
) -> OptResult:
    c = constraints or Constraints()
    n = len(mu)
    w0 = np.zeros(n) if current is None else np.asarray(current, dtype=np.float64)
    w = cp.Variable(n)
    sigma = cp.psd_wrap(np.asarray(cov, dtype=np.float64))
    trade = w - w0
    tc = np.broadcast_to(np.asarray(c.tcost, dtype=np.float64), (n,))

    objective = cp.Maximize(
        mu @ w - 0.5 * risk_aversion * cp.quad_form(w, sigma) - tc @ cp.abs(trade)
    )
    cons: list[cp.Constraint] = []
    if c.budget is not None:
        cons.append(cp.sum(w) == c.budget)
    if c.long_only:
        cons.append(w >= 0)
    if c.max_weight is not None:
        cons.append(w <= c.max_weight)
    if c.min_weight is not None:
        cons.append(w >= c.min_weight)
    if c.max_gross is not None:
        cons.append(cp.norm1(w) <= c.max_gross)
    if c.max_vol is not None:
        cons.append(cp.quad_form(w, sigma) <= c.max_vol**2)
    if c.max_turnover is not None:
        cons.append(cp.norm1(trade) <= c.max_turnover)
    for idx, cap in c.group_caps.values():
        cons.append(cp.sum(w[idx]) <= cap)

    problem = cp.Problem(objective, cons)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
        raise OptimizationError(f"optimizer status {problem.status}; constraints may be infeasible")
    weights = np.asarray(w.value, dtype=np.float64)
    weights[np.abs(weights) < 1e-9] = 0.0
    return OptResult(
        weights=weights,
        status=str(problem.status),
        expected_return=float(mu @ weights),
        volatility=float(np.sqrt(weights @ cov @ weights)),
        turnover=float(np.abs(weights - w0).sum()),
    )


def min_variance(cov: Matrix, constraints: Constraints | None = None) -> OptResult:
    """Minimum-variance portfolio: mean-variance with zero expected returns."""
    n = cov.shape[0]
    return mean_variance(np.zeros(n), cov, risk_aversion=1.0, constraints=constraints)


def risk_parity(cov: Matrix, budgets: Vector | None = None) -> Vector:
    """Equal (or budgeted) risk contribution, long-only, fully invested.

    Solved as the convex problem  min 0.5 y'Sigma y - b' log(y),  then w = y / sum(y)
    (Spinu 2013), which has a unique solution and no starting-point sensitivity.
    """
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets) / np.sum(budgets)
    y = cp.Variable(n, pos=True)
    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(cov)) - b @ cp.log(y)))
    problem.solve(solver=cp.CLARABEL)
    if y.value is None:
        raise OptimizationError(f"risk parity failed: {problem.status}")
    weights = np.asarray(y.value, dtype=np.float64)
    return weights / weights.sum()


def risk_contributions(weights: Vector, cov: Matrix) -> Vector:
    """Fraction of portfolio variance coming from each asset (sums to 1)."""
    marginal = cov @ weights
    total = float(weights @ marginal)
    return weights * marginal / total
