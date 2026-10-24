"""Stress testing: apply named shocks to the current book.

Two kinds of shock, which can be combined in one scenario:
  * instrument shocks — "EQ:AAPL": -0.30 (a direct return on that instrument)
  * factor shocks     — "SPY": -0.20, propagated to every asset through its betas

Instruments not covered by either get `default_shock`. Scenarios are data, so the
library of historical and hypothetical stresses lives in config, not code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    instrument_shocks: dict[str, float] = field(default_factory=dict)
    factor_shocks: dict[str, float] = field(default_factory=dict)
    default_shock: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class StressResult:
    scenario: str
    total_pnl: float
    pnl_by_position: pd.Series


def run_stress(
    dollar_exposures: pd.Series,
    scenario: Scenario,
    betas: pd.DataFrame | None = None,
) -> StressResult:
    shocks = pd.Series(scenario.default_shock, index=dollar_exposures.index, dtype=float)
    if scenario.factor_shocks:
        if betas is None:
            raise ValueError(f"scenario {scenario.name!r} has factor shocks but no betas given")
        f = pd.Series(scenario.factor_shocks, dtype=float)
        b = betas.reindex(index=dollar_exposures.index, columns=f.index).fillna(0.0)
        covered = b.abs().sum(axis=1) > 0
        shocks[covered] = (b @ f)[covered]
    for inst, shock in scenario.instrument_shocks.items():
        if inst in shocks.index:
            shocks[inst] = shock
    pnl = dollar_exposures * shocks
    return StressResult(scenario.name, float(pnl.sum()), pnl)


def run_stress_suite(
    dollar_exposures: pd.Series,
    scenarios: list[Scenario],
    betas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = [run_stress(dollar_exposures, s, betas) for s in scenarios]
    return pd.DataFrame(
        {"scenario": [r.scenario for r in rows], "pnl": [r.total_pnl for r in rows]}
    ).sort_values("pnl")
