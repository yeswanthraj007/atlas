"""Walk-forward validation: tune on the past, score on the unseen future, repeat.

For each fold:
  1. run every parameter set on the TRAIN window and pick the best by the objective;
  2. run only that choice on the TEST window, which it has never seen;
  3. keep the test-window returns. Stitched together they form the out-of-sample record,
     the only performance number that counts.

The holdout is evaluated separately, once, with `evaluate_holdout`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.engine import BacktestConfig, performance_metrics, run_backtest
from atlas.refdata.instruments import SecurityMaster
from atlas.strategy.base import Strategy
from atlas.validation.scorecard import round_trips
from atlas.validation.splits import SplitPlan
from atlas.validation.stats import deflated_sharpe, sharpe

Params = dict[str, Any]


@dataclass
class FoldResult:
    number: int
    chosen: Params
    train_sharpe: float  # annualised, of the chosen params on the train window
    test_sharpe: float  # annualised, out of sample
    test_return: float
    test_equity: pd.Series
    trips: pd.DataFrame
    trial_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    oos_returns: pd.Series
    n_trials: int
    trial_daily_sharpes: list[float]

    def fold_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": f.number,
                    "test_start": f.test_equity.index[0].date(),
                    "test_end": f.test_equity.index[-1].date(),
                    "chosen": ", ".join(f"{k}={v}" for k, v in f.chosen.items()),
                    "train_sharpe": f.train_sharpe,
                    "test_sharpe": f.test_sharpe,
                    "test_return": f.test_return,
                }
                for f in self.folds
            ]
        ).set_index("fold")

    def oos_equity(self, start_value: float = 1.0) -> pd.Series:
        return start_value * (1 + self.oos_returns).cumprod()

    def summary(self) -> dict[str, float]:
        r = self.oos_returns.to_numpy()
        tbl = self.fold_table()
        is_sr, oos_sr = tbl["train_sharpe"].mean(), tbl["test_sharpe"].mean()
        return {
            **performance_metrics(self.oos_equity()),
            "mean_train_sharpe": float(is_sr),
            "mean_test_sharpe": float(oos_sr),
            # Below ~0.5 the train-window performance was mostly fitting noise.
            "oos_to_is_sharpe_ratio": float(oos_sr / is_sr) if is_sr else np.nan,
            "profitable_folds": float((tbl["test_return"] > 0).mean()),
            "deflated_sharpe_prob": deflated_sharpe(
                r, self.n_trials, np.array(self.trial_daily_sharpes)
            ),
        }


def _slice(
    bars: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    out = {k: v.loc[start:end] for k, v in bars.items()}
    return {k: v for k, v in out.items() if len(v)}


def _key(p: Params) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(p.items()))


def run_window(
    make_strategy: Callable[[Params], Strategy],
    params: Params,
    bars: dict[str, pd.DataFrame],
    secmaster: SecurityMaster,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config_fn: Callable[[], BacktestConfig],
    membership: pd.DataFrame | None,
    warmup_days: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Backtest one window. History before `start` is warm-up only (no trading)."""
    window = _slice(bars, start - pd.Timedelta(days=warmup_days), end)
    res = run_backtest(
        make_strategy(params),
        window,
        secmaster,
        config_fn(),
        membership=membership,
        start_trading=start,
    )
    equity = res.equity.loc[start:end]
    marks = {i: Decimal(str(round(df["close"].iloc[-1], 6))) for i, df in window.items()}
    return equity, round_trips(res.ledger.transactions(), marks)


def walk_forward(
    make_strategy: Callable[[Params], Strategy],
    grid: list[Params],
    bars: dict[str, pd.DataFrame],
    secmaster: SecurityMaster,
    plan: SplitPlan,
    config_fn: Callable[[], BacktestConfig],
    membership: pd.DataFrame | None = None,
    warmup_days: int = 400,
    objective: Callable[[pd.Series], float] | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> WalkForwardResult:
    score = objective or (lambda eq: performance_metrics(eq).get("sharpe", -np.inf))
    folds: list[FoldResult] = []
    trial_srs: list[float] = []
    for fold in plan.folds:
        trial_scores: dict[str, float] = {}
        best, best_score = grid[0], -np.inf
        for params in grid:
            eq, _ = run_window(
                make_strategy,
                params,
                bars,
                secmaster,
                fold.train_start,
                fold.train_end,
                config_fn,
                membership,
                warmup_days,
            )
            s = float(score(eq))
            trial_scores[_key(params)] = s
            trial_srs.append(sharpe(eq.pct_change().dropna().to_numpy()))
            if s > best_score:
                best, best_score = params, s
        test_eq, trips = run_window(
            make_strategy,
            best,
            bars,
            secmaster,
            fold.test_start,
            fold.test_end,
            config_fn,
            membership,
            warmup_days,
        )
        m = performance_metrics(test_eq)
        folds.append(
            FoldResult(
                fold.number,
                best,
                best_score,
                m.get("sharpe", np.nan),
                m.get("total_return", np.nan),
                test_eq,
                trips,
                trial_scores,
            )
        )
        progress(
            f"{fold} -> {_key(best)}: train SR {best_score:.2f}, "
            f"test SR {m.get('sharpe', np.nan):.2f}"
        )
    oos = pd.concat([f.test_equity.pct_change().dropna() for f in folds]).sort_index()
    return WalkForwardResult(folds, oos, len(grid), trial_srs)


def evaluate_holdout(
    make_strategy: Callable[[Params], Strategy],
    params: Params,
    bars: dict[str, pd.DataFrame],
    secmaster: SecurityMaster,
    plan: SplitPlan,
    config_fn: Callable[[], BacktestConfig],
    membership: pd.DataFrame | None = None,
    warmup_days: int = 400,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run the final choice on the locked holdout. Do this ONCE, after all development:
    looking at the holdout and then changing the strategy turns it into training data."""
    if plan.holdout_start is None or plan.holdout_end is None:
        raise ValueError("plan has no holdout")
    return run_window(
        make_strategy,
        params,
        bars,
        secmaster,
        plan.holdout_start,
        plan.holdout_end,
        config_fn,
        membership,
        warmup_days,
    )
