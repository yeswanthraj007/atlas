"""Volatility-managed strategies driven by forecast (not realised) volatility.

Moreira & Muir (2017): scaling exposure inversely to expected variance has historically
raised risk-adjusted returns, because volatility is predictable but returns do not rise
proportionally when volatility spikes. Here exposure is capped at 1.0 (no borrowing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas.core.types import ZERO
from atlas.strategy.base import StrategyContext, TargetWeights


def exposure_from_forecast(forecast_vol: float, target_vol: float, cap: float) -> float:
    if not np.isfinite(forecast_vol) or forecast_vol <= 0:
        return cap
    return float(min(cap, target_vol / forecast_vol))


def _current_weight(ctx: StrategyContext, inst: str) -> float:
    px = ctx.prices.get(inst)
    if px is None or ctx.nav <= ZERO:
        return 0.0
    return float(ctx.book.quantity(inst) * px * ctx.multiplier(inst) / ctx.nav)


@dataclass
class VolTargetIndex:
    """Hold one index instrument at exposure = target_vol / forecast_vol (capped)."""

    instrument: str
    forecast: pd.Series  # forecast annualised vol, known at each date's close
    target_vol: float
    cap: float = 1.0
    gross: float = 0.98
    rebalance_every: int = 5
    band: float = 0.05  # skip trades smaller than 5% of NAV (cost control)
    id: str = "vol_target_index"

    def on_bar(self, ctx: StrategyContext) -> TargetWeights | None:
        if ctx.bar_index % self.rebalance_every != 0:
            return None
        f = self.forecast.loc[: ctx.timestamp].dropna()
        if f.empty:
            return None
        w = self.gross * exposure_from_forecast(float(f.iloc[-1]), self.target_vol, self.cap)
        if ctx.bar_index > 0 and abs(w - _current_weight(ctx, self.instrument)) < self.band:
            return None
        return {self.instrument: w}


@dataclass
class RiskBalancedStocks:
    """Weights proportional to 1 / forecast stock volatility, total exposure scaled by the
    market volatility forecast. Each stock then contributes roughly equal standalone risk."""

    stock_forecast: pd.DataFrame  # dates x instruments, forecast annualised vol
    market_forecast: pd.Series
    target_vol: float
    cap: float = 1.0
    gross: float = 0.98
    max_weight: float = 0.02
    rebalance_every: int = 21
    id: str = "risk_balanced"

    def on_bar(self, ctx: StrategyContext) -> TargetWeights | None:
        if ctx.bar_index % self.rebalance_every != 0:
            return None
        ts = ctx.timestamp
        sf = self.stock_forecast.loc[:ts]
        mf = self.market_forecast.loc[:ts].dropna()
        if sf.empty or mf.empty:
            return None
        vol = sf.iloc[-1].dropna()
        if ctx.eligible is not None:
            vol = vol[vol.index.isin(ctx.eligible)]
        vol = vol[vol.index.isin(ctx.prices.keys()) & (vol > 0)]
        if vol.empty:
            return None
        inv = 1.0 / vol
        exposure = self.gross * exposure_from_forecast(
            float(mf.iloc[-1]), self.target_vol, self.cap
        )
        w = np.minimum(inv / inv.sum() * exposure, self.max_weight)
        return {k: float(v) for k, v in w.items() if v > 0}


@dataclass
class EqualWeightStocks:
    rebalance_every: int = 21
    gross: float = 0.98
    id: str = "equal_weight"

    def on_bar(self, ctx: StrategyContext) -> TargetWeights | None:
        if ctx.bar_index % self.rebalance_every != 0:
            return None
        names = [k for k in ctx.prices if ctx.eligible is None or k in ctx.eligible]
        names = [k for k in names if k.startswith("EQ:")]
        return {n: self.gross / len(names) for n in names} if names else None
