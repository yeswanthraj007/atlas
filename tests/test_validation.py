from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.engine import BacktestConfig
from atlas.execution.broker import CostModel
from atlas.ledger.ledger import Transaction, TxnKind
from atlas.strategy.momentum import CrossSectionalMomentum
from atlas.validation import scorecard as sc
from atlas.validation.signal import (
    forward_returns,
    information_coefficient,
    per_ticker_signal_scores,
    signal_panel,
    summarize_ic,
)
from atlas.validation.splits import walk_forward_splits
from atlas.validation.stats import (
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)
from atlas.validation.walkforward import evaluate_holdout, walk_forward
from tests.test_oms_backtest import gbm_bars

RNG = np.random.default_rng(11)
DATES = pd.bdate_range("2016-01-04", "2026-10-23", tz="UTC")


def test_walk_forward_splits_never_leak() -> None:
    plan = walk_forward_splits(DATES, DATES[0], 756, 252, embargo_days=21, holdout_days=252)
    assert len(plan.folds) >= 5
    prev_test_end = None
    for f in plan.folds:
        assert f.train_start == DATES[0]  # expanding window
        assert f.train_end < f.test_start
        gap = DATES.get_loc(f.test_start) - DATES.get_loc(f.train_end) - 1
        assert gap == 21  # embargo
        if prev_test_end is not None:  # test windows are back to back
            assert DATES.get_loc(f.test_start) == DATES.get_loc(prev_test_end) + 1
        prev_test_end = f.test_end
        assert f.test_end < plan.holdout_start
    assert plan.holdout_end == DATES[-1]
    assert DATES.get_loc(plan.holdout_start) - DATES.get_loc(plan.folds[-1].test_end) - 1 == 21
    rolling = walk_forward_splits(DATES, DATES[0], 756, 252, expanding=False)
    assert rolling.folds[2].train_start > DATES[0]
    with pytest.raises(ValueError):
        walk_forward_splits(DATES[:300], DATES[0], 756, 252)


def test_sharpe_statistics() -> None:
    noise = RNG.normal(0, 0.01, 2000)
    assert 0.2 < probabilistic_sharpe(noise) < 0.8  # no skill: coin flip
    skilled = RNG.normal(0.002, 0.01, 2000)  # daily SR 0.2, annual ~3.2
    assert probabilistic_sharpe(skilled) > 0.999
    assert expected_max_sharpe(100, 0.001) > expected_max_sharpe(10, 0.001) > 0
    modest = RNG.normal(0.0004, 0.01, 1000)  # annual SR ~0.6
    assert deflated_sharpe(modest, 1) > deflated_sharpe(modest, 200)


def test_information_coefficient_detects_skill_and_noise() -> None:
    idx = pd.bdate_range("2024-01-01", periods=400, tz="UTC")
    rets = pd.DataFrame(
        RNG.normal(0, 0.01, (400, 30)), index=idx, columns=[f"S{i}" for i in range(30)]
    )
    closes = 100 * (1 + rets).cumprod()
    dates = idx[::21][:-2]
    fwd = forward_returns(closes, dates, 21)
    perfect = fwd.copy()  # a "signal" equal to the future: IC = 1 (and is look-ahead)
    assert summarize_ic(information_coefficient(perfect, fwd)).mean_ic == pytest.approx(1.0)
    noise = pd.DataFrame(RNG.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    s = summarize_ic(information_coefficient(noise, fwd))
    assert abs(s.mean_ic) < 0.1 and abs(s.t_stat) < 3
    scores = per_ticker_signal_scores(perfect, fwd, top_n=5)
    assert (scores["hit_rate_when_selected"].dropna() == 1.0).all()


def test_signal_panel_only_sees_the_past() -> None:
    idx = pd.bdate_range("2024-01-01", periods=60, tz="UTC")
    closes = pd.DataFrame({"A": np.arange(60.0) + 1, "B": np.arange(60.0) + 2}, index=idx)

    def fn(hist: pd.DataFrame) -> pd.Series:
        fn.last_seen.append(hist.index[-1])
        return hist.iloc[-1]

    fn.last_seen = []
    dates = idx[[10, 30, 50]]
    signal_panel(closes, fn, dates)
    assert list(fn.last_seen) == list(dates)


def _t(day: int, qty: str, px: str) -> Transaction:
    ts = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day)
    return Transaction("p", TxnKind.TRADE, D(qty), ts, "EQ:A", D(px), recorded_at=ts)


def test_round_trips_and_scorecard() -> None:
    txns = [
        _t(0, "10", "100"),
        _t(5, "-10", "110"),  # +100, win
        _t(10, "5", "100"),
        _t(20, "-5", "90"),  # -50, loss
        _t(30, "4", "50"),
    ]  # still open, marked at 60: +40
    trips = sc.round_trips(txns, {"EQ:A": D(60)})
    assert list(trips["pnl"]) == [100.0, -50.0, 40.0]
    assert list(trips["open_at_end"]) == [False, False, True]
    assert trips["return"].iloc[0] == pytest.approx(0.1)
    card = sc.ticker_scorecard(trips)
    assert card.loc["EQ:A", "trips"] == 3
    assert card.loc["EQ:A", "win_rate"] == pytest.approx(2 / 3)


def test_drawdown_and_yearly_tables() -> None:
    idx = pd.bdate_range("2024-01-01", periods=8, tz="UTC")
    eq = pd.Series([100, 110, 99, 88, 110, 120, 108, 125.0], index=idx)
    dd = sc.drawdown_table(eq)
    assert dd["depth"].iloc[0] == pytest.approx(88 / 110 - 1)
    assert dd["trough"].iloc[0] == idx[3] and dd["recovered"].iloc[0] == idx[4]
    yearly = sc.yearly_table(eq, eq * 1.0)
    assert yearly["excess"].abs().max() < 1e-12


def test_regime_table_uses_only_prior_information() -> None:
    idx = pd.bdate_range("2020-01-01", periods=600, tz="UTC")
    bench = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0.0003, 0.01, 600))), index=idx)
    strat = bench.pct_change()
    tbl = sc.regime_table(strat, bench)
    assert set(tbl.index) == {"bull_trend", "bear_trend", "low_vol", "mid_vol", "high_vol"}
    assert tbl.loc[["low_vol", "mid_vol", "high_vol"], "days"].sum() <= len(strat.dropna())
    # strategy == benchmark, so per-regime returns must match the benchmark column
    assert tbl["ann_return"].to_numpy() == pytest.approx(tbl["benchmark_ann_return"].to_numpy())


def test_walk_forward_end_to_end_is_out_of_sample() -> None:
    bars, sm = gbm_bars(n_assets=12, n_days=900, seed=3)
    dates = next(iter(bars.values())).index
    plan = walk_forward_splits(dates, dates[0], 300, 120, embargo_days=21, holdout_days=100)

    def cfg() -> BacktestConfig:
        return BacktestConfig(costs=CostModel(per_share=D(0), min_per_order=D(0)))

    grid = [
        {"lookback": 60, "skip": 5, "top_n": 3, "rebalance_every": 10},
        {"lookback": 120, "skip": 5, "top_n": 3, "rebalance_every": 10},
    ]
    res = walk_forward(
        lambda p: CrossSectionalMomentum(**p), grid, bars, sm, plan, cfg, warmup_days=250
    )
    assert len(res.folds) == len(plan.folds)
    for f, fold in zip(res.folds, plan.folds, strict=True):
        assert f.test_equity.index[0] == fold.test_start
        assert f.test_equity.index[-1] == fold.test_end
        assert f.chosen in grid
    assert res.oos_returns.index.is_unique
    assert res.oos_returns.index.max() < plan.holdout_start
    summary = res.summary()
    assert 0.0 <= summary["deflated_sharpe_prob"] <= 1.0
    hold_eq, _ = evaluate_holdout(
        lambda p: CrossSectionalMomentum(**p), grid[0], bars, sm, plan, cfg, warmup_days=250
    )
    assert hold_eq.index[0] == plan.holdout_start


def test_alpha_beta_separates_skill_from_leverage() -> None:
    from atlas.validation.stats import alpha_beta

    mkt = RNG.normal(0.0005, 0.01, 3000)
    levered = 1.5 * mkt + RNG.normal(0, 0.002, 3000)  # beats the market only via beta
    ab = alpha_beta(levered, mkt)
    assert ab["beta"] == pytest.approx(1.5, abs=0.02)
    assert abs(ab["alpha_t"]) < 2.5
    skilled = mkt + 0.0004 + RNG.normal(0, 0.002, 3000)  # ~10%/yr true alpha
    assert alpha_beta(skilled, mkt)["alpha_t"] > 4
