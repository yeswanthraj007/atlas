"""VaR model backtests. A risk number nobody has backtested is an opinion.

* Kupiec (1995) proportion-of-failures: is the exception *rate* right?
* Christoffersen (1998) independence: do exceptions cluster (model too slow to react)?
* Conditional coverage = both at once.
* Basel traffic light on the last 250 days.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float
    passed: bool  # at the chosen significance level


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def exceptions(pnl: NDArray[np.float64], var: NDArray[np.float64]) -> NDArray[np.bool_]:
    """True where the realised loss exceeded the VaR forecast made the day before."""
    return -np.asarray(pnl) > np.asarray(var)


def kupiec_pof(
    hits: NDArray[np.bool_], confidence: float, significance: float = 0.05
) -> TestResult:
    n = int(hits.size)
    x = int(np.sum(hits))
    p = 1.0 - confidence
    phat = x / n
    log_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    log_alt = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = -2.0 * (log_null - log_alt)
    pval = float(stats.chi2.sf(lr, df=1))
    return TestResult("kupiec_pof", float(lr), pval, pval >= significance)


def christoffersen_independence(hits: NDArray[np.bool_], significance: float = 0.05) -> TestResult:
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    log_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    log_alt = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr = max(-2.0 * (log_null - log_alt), 0.0)
    pval = float(stats.chi2.sf(lr, df=1))
    return TestResult("christoffersen_independence", float(lr), pval, pval >= significance)


def conditional_coverage(
    hits: NDArray[np.bool_], confidence: float, significance: float = 0.05
) -> TestResult:
    lr = kupiec_pof(hits, confidence).statistic + christoffersen_independence(hits).statistic
    pval = float(stats.chi2.sf(lr, df=2))
    return TestResult("conditional_coverage", lr, pval, pval >= significance)


class TrafficLight(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


def basel_traffic_light(hits: NDArray[np.bool_]) -> TrafficLight:
    """Basel zones for a 99% VaR over the most recent 250 observations."""
    x = int(np.sum(np.asarray(hits)[-250:]))
    if x <= 4:
        return TrafficLight.GREEN
    if x <= 9:
        return TrafficLight.YELLOW
    return TrafficLight.RED
