from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.engine import BacktestConfig
from atlas.execution.broker import CostModel
from atlas.ml.dataset import build_dataset, feature_columns
from atlas.ml.models import GBMRanker, RidgeRanker, cross_validate, purged_time_splits
from atlas.ml.walkforward import ml_walk_forward, sample_dates
from atlas.validation.splits import walk_forward_splits
from tests.test_oms_backtest import gbm_bars

RNG = np.random.default_rng(3)


def synthetic(n_days=900, n=80, strength=0.01):
    """A hidden characteristic `quality` drives next-month returns (plus noise)."""
    idx = pd.bdate_range("2018-01-01", periods=n_days, tz="UTC")
    cols = [f"EQ:S{i}" for i in range(n)]
    quality = pd.DataFrame(RNG.normal(size=(n_days, n)), index=idx, columns=cols)
    quality = quality.rolling(42, min_periods=1).mean()  # persistent, like real fundamentals
    rets = strength * quality.shift(1).fillna(0) + RNG.normal(0, 0.015, (n_days, n))
    closes = 100 * np.exp(rets.cumsum())
    noise = pd.DataFrame(RNG.normal(size=(n_days, n)), index=idx, columns=cols)
    return closes, {"quality": quality, "noise": noise}


def test_dataset_targets_are_future_and_labelled() -> None:
    closes, factors = synthetic(n_days=200, n=30)
    dates = closes.index[::21][:-1]
    data = build_dataset(factors, closes, dates, horizon=21)
    assert set(feature_columns(data)) == {"quality", "noise"}
    d0, d1 = dates[0], dates[1]
    assert data.xs(d0, level="date")["label_end"].iloc[0] == closes.index[21]
    fwd = closes.loc[d1] / closes.loc[d0] - 1
    expected = (fwd.rank(pct=True) - 0.5).sort_index()
    got = data.xs(d0, level="date")["target"].sort_index()
    assert got.to_numpy() == pytest.approx(expected.to_numpy())
    assert data["quality"].between(-0.5, 0.5).all()
    last = data.xs(dates[-1], level="date")
    assert last["target"].notna().all()  # its 21-day label still fits in the data


def test_purged_splits_never_train_on_unknown_labels() -> None:
    dates = pd.bdate_range("2018-01-01", periods=60, freq="21B", tz="UTC")
    label_end = pd.Series(dates + pd.offsets.BDay(21), index=dates)
    for train, val in purged_time_splits(list(dates), label_end, n_splits=3, min_train=20):
        assert max(label_end[t] for t in train) < val[0]
        assert max(train) < min(val)


def test_models_find_planted_signal_and_not_noise() -> None:
    closes, factors = synthetic()
    data = build_dataset(factors, closes, closes.index[::21][:-1], horizon=21)
    data = data[data["target"].notna()]
    feats = feature_columns(data)
    for make in (lambda: RidgeRanker(), lambda: GBMRanker(min_samples_leaf=50, max_iter=100)):
        assert cross_validate(make, data, feats, n_splits=3) > 0.1
    noise_only = data.drop(columns=["quality"])
    ic = cross_validate(lambda: RidgeRanker(), noise_only, ["noise"], n_splits=3)
    assert abs(ic) < 0.06


def test_ml_walk_forward_trains_only_on_the_past() -> None:
    bars, sm = gbm_bars(n_assets=25, n_days=700, seed=9)
    closes = pd.DataFrame({k: v["close"] for k, v in bars.items()})
    quality = closes.pct_change(63).rank(axis=1)  # any causal feature works here
    plan = walk_forward_splits(closes.index, closes.index[0], 300, 120, 21, holdout_days=60)
    data = build_dataset({"q": quality}, closes, sample_dates(closes.index, plan), horizon=21)
    fits: list[tuple[int, pd.Timestamp]] = []
    fold_no = [0]
    label_end = data.groupby(level="date")["label_end"].first()

    class Spy(RidgeRanker):
        def fit(self, x, y):
            fits.append((fold_no[0], x.index.get_level_values("date").max()))
            return super().fit(x, y)

    def cfg() -> BacktestConfig:
        return BacktestConfig(costs=CostModel(per_share=D(0), min_per_order=D(0)))

    def progress(_msg: str) -> None:
        fold_no[0] += 1

    res = ml_walk_forward(
        data,
        {"ridge": lambda: Spy()},
        plan,
        bars,
        closes,
        sm,
        cfg,
        top_n=5,
        warmup_days=200,
        progress=progress,
    )
    assert len(res.wf.folds) == len(plan.folds)
    assert {k for k, _ in fits} == set(range(len(plan.folds)))
    for k, latest in fits:  # every fit only saw outcomes known before its fold's cutoff
        assert label_end[latest] <= plan.folds[k].train_end
    for d in res.predictions.index:
        assert any(f.test_start <= d <= f.test_end for f in plan.folds)
    assert res.predictions.index.max() < plan.holdout_start
