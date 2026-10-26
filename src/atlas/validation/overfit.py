"""Overfitting and statistical-power diagnostics.

* `pbo_cscv`: Probability of Backtest Overfitting via combinatorially symmetric
  cross-validation (Bailey, Borwein, Lopez de Prado & Zhu 2015). Split time into S blocks;
  for every way of choosing half the blocks as "in sample", pick the variant with the best
  in-sample Sharpe and see where it ranks on the other half. PBO is the share of splits in
  which the in-sample winner falls below the out-of-sample median. Near 0.5 or above: the
  selection process is picking noise.
* `min_detectable_ic`: the smallest mean IC a test of n independent periods can
  distinguish from zero at a given t-stat, for the observed IC volatility. A "failed" test
  of a signal weaker than this says nothing either way.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray


def _sharpe(x: NDArray[np.float64]) -> NDArray[np.float64]:
    sd = x.std(axis=0, ddof=1)
    return np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1.0), 0.0)


def pbo_cscv(returns: NDArray[np.float64], n_blocks: int = 16) -> dict[str, float]:
    """`returns`: periods x variants (one column per parameter setting tried)."""
    r = np.asarray(returns, dtype=np.float64)
    t, n = r.shape
    if n < 2:
        raise ValueError("need at least two variants to measure selection overfitting")
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks must be an even number >= 4")
    usable = (t // n_blocks) * n_blocks
    if usable < n_blocks * 3:
        raise ValueError("not enough periods for this many blocks")
    blocks = np.array_split(r[-usable:], n_blocks)
    logits, degradation = [], []
    for ins in combinations(range(n_blocks), n_blocks // 2):
        is_set = set(ins)
        is_r = np.vstack([blocks[i] for i in range(n_blocks) if i in is_set])
        oos_r = np.vstack([blocks[i] for i in range(n_blocks) if i not in is_set])
        is_sr, oos_sr = _sharpe(is_r), _sharpe(oos_r)
        best = int(np.argmax(is_sr))
        rank = (oos_sr < oos_sr[best]).sum() + 0.5 * ((oos_sr == oos_sr[best]).sum() - 1)
        omega = (rank + 1) / (n + 1)
        logits.append(np.log(omega / (1 - omega)))
        degradation.append(oos_sr[best] - is_sr[best])
    logits_a = np.array(logits)
    return {
        "pbo": float((logits_a <= 0).mean()),
        "splits": len(logits),
        "mean_sharpe_degradation": float(np.mean(degradation)),
    }


def min_detectable_ic(n_periods: int, ic_std: float, t_required: float = 2.0) -> float:
    """Smallest mean IC detectable at `t_required` with n independent periods."""
    if n_periods < 2:
        return float("inf")
    return float(t_required * ic_std / np.sqrt(n_periods))


def periods_needed(ic: float, ic_std: float, t_required: float = 2.0) -> int:
    """Independent periods needed for a true mean IC to reach `t_required` on average."""
    if ic == 0:
        return int(1e9)
    return int(np.ceil((t_required * ic_std / abs(ic)) ** 2))
