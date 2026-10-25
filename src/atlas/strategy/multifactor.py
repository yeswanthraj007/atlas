"""Multi-factor equity strategy.

On each rebalance date:
  1. read every factor for the eligible stocks from the precomputed `FactorPanel`;
  2. winsorize, z-score and (optionally) sector-neutralise each factor;
  3. combine them, either equally or weighted by each factor's trailing IC (only IC
     observations whose forward window has already closed);
  4. hold the top `top_n` by composite score, equal or inverse-volatility weighted;
  5. optionally scale exposure down when the benchmark is below its 200-day average.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from atlas.factors.combine import composite, ic_weights, standardize
from atlas.factors.panel import FactorPanel
from atlas.strategy.base import StrategyContext, TargetWeights


@dataclass
class MultiFactor:
    panel: FactorPanel
    factors: tuple[str, ...] = ("vol_adj_mom", "reversal_1m", "low_vol", "high_52w")
    weighting: str = "ic"  # "ic" or "equal"
    top_n: int = 30
    rebalance_every: int = 21
    sectors: Mapping[str, str] | None = None
    sector_neutral: bool = True
    sizing: str = "inverse_vol"  # or "equal"
    max_weight: float = 0.06
    gross: float = 0.98
    ic_window: int = 24  # past rebalance periods used for IC weights (~2 years)
    trend_filter: bool = False
    benchmark: pd.Series | None = None  # benchmark closes, for the trend filter
    defensive_gross: float = 0.5
    id: str = "multifactor"
    last_weights: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        unknown = set(self.factors) - set(self.panel.factors)
        if unknown:
            raise ValueError(f"factors not in panel: {sorted(unknown)}")
        if self.trend_filter and self.benchmark is None:
            raise ValueError("trend_filter needs benchmark closes")

    def factor_weights(self, ts: pd.Timestamp) -> dict[str, float]:
        if self.weighting == "equal":
            return dict.fromkeys(self.factors, 1.0)
        hist = self.panel.ic_history(ts, self.ic_window)[list(self.factors)]
        return ic_weights(hist)

    def scores(self, ts: pd.Timestamp, eligible: frozenset[str] | None) -> pd.Series:
        raw = self.panel.at(ts, eligible)
        sectors = self.sectors if self.sector_neutral else None
        z = {f: standardize(raw[f], sectors) for f in self.factors}
        self.last_weights = self.factor_weights(ts)
        return composite(z, self.last_weights, min_factors=max(1, len(self.factors) - 1))

    def _exposure(self, ts: pd.Timestamp) -> float:
        if not self.trend_filter or self.benchmark is None:
            return self.gross
        b = self.benchmark.loc[:ts].dropna()
        if len(b) < 200:
            return self.gross
        return self.gross if b.iloc[-1] >= b.tail(200).mean() else self.defensive_gross

    def on_bar(self, ctx: StrategyContext) -> TargetWeights | None:
        if ctx.bar_index % self.rebalance_every != 0:
            return None
        ts = pd.Timestamp(ctx.timestamp)
        if ts not in self.panel.closes.index:
            return None
        score = self.scores(ts, ctx.eligible)
        score = score[score.index.isin(ctx.prices.keys())]
        if score.empty:
            return None
        picks = score.nlargest(self.top_n).index
        exposure = self._exposure(ts)
        if self.sizing == "inverse_vol":
            vol = -self.panel.raw["low_vol"].loc[ts, picks]
            inv = 1.0 / vol.where(vol > 0)
            w = (inv / inv.sum()).fillna(0.0)
        else:
            w = pd.Series(1.0 / len(picks), index=picks)
        w = np.minimum(w * exposure, self.max_weight)
        return {k: float(v) for k, v in w.items() if v > 0}
