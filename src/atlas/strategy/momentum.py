"""Cross-sectional momentum with risk-aware sizing — the reference systematic strategy.

Signal: 12-month return skipping the most recent month (Jegadeesh & Titman 1993), which
avoids the short-term reversal effect. Each rebalance holds the top `top_n` names.
Sizing is either inverse-volatility or the mean-variance optimizer with a volatility cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas.optimizer.mean_variance import Constraints, OptimizationError, mean_variance
from atlas.risk.covariance import ledoit_wolf
from atlas.strategy.base import StrategyContext, TargetWeights


@dataclass
class CrossSectionalMomentum:
    id: str = "xs_momentum"
    lookback: int = 252
    skip: int = 21
    top_n: int = 5
    rebalance_every: int = 21
    sizing: str = "inverse_vol"  # or "optimizer"
    max_weight: float = 0.25
    gross: float = 0.98  # keep a little cash for costs
    annual_vol_target: float | None = 0.15

    def signal(self, history: pd.DataFrame) -> pd.Series | None:
        """Momentum score per instrument at the last row of `history` (higher = stronger).

        Only names with a complete price history over the lookback window qualify.
        Exposed separately so the signal can be scored on its own (information coefficient).
        """
        closes = history.tail(self.lookback + 1).dropna(axis=1, how="any")
        if len(closes) <= self.lookback or closes.shape[1] == 0:
            return None
        return closes.iloc[-1 - self.skip] / closes.iloc[0] - 1.0

    def on_bar(self, ctx: StrategyContext) -> TargetWeights | None:
        if ctx.bar_index % self.rebalance_every != 0:
            return None
        signal = self.signal(ctx.tradable_history())
        if signal is None:
            return None
        closes = ctx.history[signal.index].tail(self.lookback + 1)
        winners = signal[signal > 0].nlargest(self.top_n).index.tolist()
        if not winners:
            return {}  # nothing trending up: go to cash
        rets = closes[winners].pct_change().dropna().tail(126).to_numpy()
        if self.sizing == "optimizer" and len(winners) > 1:
            return self._optimize(winners, rets, signal[winners].to_numpy())
        vol = rets.std(axis=0, ddof=1)
        inv = 1.0 / np.where(vol > 0, vol, np.nan)
        w = np.nan_to_num(inv / np.nansum(inv)) * self.gross
        w = np.minimum(w, self.max_weight)
        return dict(zip(winners, w.tolist(), strict=True))

    def _optimize(self, names: list[str], rets: np.ndarray, score: np.ndarray) -> TargetWeights:
        cov, _ = ledoit_wolf(rets)
        mu = score / 252.0  # crude daily expected-return proxy from the signal
        daily_cap = (
            None if self.annual_vol_target is None else self.annual_vol_target / np.sqrt(252)
        )
        cons = Constraints(
            budget=None,
            long_only=True,
            max_weight=self.max_weight,
            max_gross=self.gross,
            max_vol=daily_cap,
        )
        try:
            res = mean_variance(mu, cov, risk_aversion=5.0, constraints=cons)
        except OptimizationError:
            return {}
        return {n: float(w) for n, w in zip(names, res.weights, strict=True) if w > 0}
