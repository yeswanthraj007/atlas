"""Walk-forward training of the return-ranking model, and its portfolio backtest.

Per fold:
  1. training rows = sample dates whose label (forward return) was KNOWN by train_end;
  2. each candidate model configuration is scored by purged time-series CV on those
     rows only; the best mean IC wins (every configuration counts as a trial);
  3. the winner is refit on all training rows and predicts the fold's test dates;
  4. predictions become the `ml_score` factor and a fixed portfolio rule trades them
     through the production OMS on the test window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.engine import BacktestConfig, performance_metrics
from atlas.factors.panel import FactorPanel
from atlas.ml.dataset import feature_columns
from atlas.ml.models import cross_validate, daily_ic
from atlas.refdata.instruments import SecurityMaster
from atlas.strategy.multifactor import MultiFactor
from atlas.validation.splits import SplitPlan
from atlas.validation.walkforward import FoldResult, WalkForwardResult, run_window


@dataclass
class MLRunResult:
    wf: WalkForwardResult
    predictions: pd.DataFrame  # test rebalance date x instrument
    cv_table: pd.DataFrame  # fold x model config -> CV IC
    importance: pd.DataFrame  # fold x feature
    oos_ic: pd.Series  # per test date: IC of predictions vs realised target


def rebalance_dates(
    index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, every: int
) -> pd.DatetimeIndex:
    window = index[(index >= start) & (index <= end)]
    return window[::every]


def sample_dates(index: pd.DatetimeIndex, plan: SplitPlan, every: int = 21) -> pd.DatetimeIndex:
    """Dates the dataset must cover: a regular training grid plus every fold's (and the
    holdout's) test rebalance dates, which start on each window's first day."""
    grid = index[::every]
    extra = [rebalance_dates(index, f.test_start, f.test_end, every) for f in plan.folds]
    if plan.holdout_start is not None and plan.holdout_end is not None:
        extra.append(rebalance_dates(index, plan.holdout_start, plan.holdout_end, every))
    return grid.union(pd.DatetimeIndex(np.concatenate([e.to_numpy() for e in extra])))


def ml_walk_forward(
    data: pd.DataFrame,
    configs: dict[str, Callable[[], Any]],
    plan: SplitPlan,
    bars: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    secmaster: SecurityMaster,
    config_fn: Callable[[], BacktestConfig],
    membership: pd.DataFrame | None = None,
    sectors: dict[str, str] | None = None,
    top_n: int = 30,
    every: int = 21,
    warmup_days: int = 400,
    progress: Callable[[str], None] = lambda _: None,
) -> MLRunResult:
    feats = feature_columns(data)
    dates = data.index.get_level_values("date")
    label_end = pd.to_datetime(data["label_end"])
    pred_rows: dict[pd.Timestamp, pd.Series] = {}
    folds: list[FoldResult] = []
    cv_rows, imp_rows, oos_ic = {}, {}, []
    trial_srs: list[float] = []

    base_panel = FactorPanel.from_closes(closes)
    for fold in plan.folds:
        train = data[(label_end <= fold.train_end) & data["target"].notna()]
        scores = {name: cross_validate(make, train, feats) for name, make in configs.items()}
        cv_rows[fold.number] = scores
        best = max(scores, key=lambda k: -np.inf if np.isnan(scores[k]) else scores[k])
        model = configs[best]().fit(train[feats], train["target"])
        imp = (
            model.importance(
                train[feats].sample(min(20_000, len(train)), random_state=0),
                train["target"].sample(min(20_000, len(train)), random_state=0),
            )
            if hasattr(model, "importance") and best.startswith("gbm")
            else model.importance()
        )
        imp_rows[fold.number] = imp

        test_dates = rebalance_dates(closes.index, fold.test_start, fold.test_end, every)
        test = data[dates.isin(test_dates)]
        missing = set(test_dates) - set(test.index.get_level_values("date"))
        if missing:
            raise ValueError(
                f"dataset lacks {len(missing)} test rebalance dates of fold {fold.number}; "
                "build it on sample_dates(...)"
            )
        pred = pd.Series(model.predict(test[feats]), index=test.index)
        for d, g in pred.groupby(level="date"):
            pred_rows[d] = g.droplevel("date")
        ic = daily_ic(pred, test["target"])
        oos_ic.append(ic)

        # trade the predictions on the test window
        panel = FactorPanel(base_panel.closes, dict(base_panel.raw))
        frame = pd.DataFrame({d: pred_rows[d] for d in test_dates if d in pred_rows}).T
        panel.add("ml_score", frame)
        params = {"model": best, "top_n": top_n, "rebalance_every": every}

        def make_strategy(p: dict, panel: FactorPanel = panel) -> MultiFactor:
            return MultiFactor(
                panel,
                factors=("ml_score",),
                weighting="equal",
                top_n=p["top_n"],
                rebalance_every=p["rebalance_every"],
                sectors=sectors,
                sector_neutral=False,
            )

        eq, trips = run_window(
            make_strategy,
            params,
            bars,
            secmaster,
            fold.test_start,
            fold.test_end,
            config_fn,
            membership,
            warmup_days,
        )
        m = performance_metrics(eq)
        folds.append(
            FoldResult(
                fold.number,
                params,
                float("nan"),
                m.get("sharpe", np.nan),
                m.get("total_return", np.nan),
                eq,
                trips,
                {k: float(v) for k, v in scores.items()},
            )
        )
        trial_srs.append(m.get("sharpe", 0.0) / np.sqrt(252))
        progress(
            f"{fold} -> {best} (CV IC {scores[best]:.3f}): test IC {ic.mean():.3f}, "
            f"test Sharpe {m.get('sharpe', np.nan):.2f}"
        )

    oos = pd.concat([f.test_equity.pct_change().dropna() for f in folds]).sort_index()
    wf = WalkForwardResult(folds, oos, len(configs), trial_srs)
    return MLRunResult(
        wf,
        pd.DataFrame(pred_rows).T.sort_index(),
        pd.DataFrame(cv_rows).T,
        pd.DataFrame(imp_rows).T,
        pd.concat(oos_ic).sort_index(),
    )
