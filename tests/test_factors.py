from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from atlas.factors import price
from atlas.factors.combine import composite, ic_weights, sector_neutral, standardize, winsorize
from atlas.factors.panel import FactorPanel, forward_returns_daily, price_factor_frames
from atlas.ledger.ledger import PositionBook
from atlas.strategy.base import StrategyContext
from atlas.strategy.multifactor import MultiFactor

RNG = np.random.default_rng(21)


def closes_panel(n_days: int = 600, n: int = 40, late: int = 3) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n_days, tz="UTC")
    rets = RNG.normal(0.0003, 0.015, (n_days, n))
    df = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"EQ:S{i}" for i in range(n)]
    )
    df.iloc[:300, :late] = np.nan  # a few late listings
    return df


def test_vectorised_panel_matches_reference_functions() -> None:
    c = closes_panel()
    frames = price_factor_frames(c)
    for pos in [260, 400, 599]:
        ts = c.index[pos]
        hist = c.loc[:ts]
        for name, fn in price.PRICE_FACTORS.items():
            ref = fn(hist).sort_index()
            got = frames[name].loc[ts].dropna().sort_index()
            assert list(ref.index) == list(got.index), (name, pos)
            assert got.to_numpy() == pytest.approx(ref.to_numpy(), rel=1e-9), (name, pos)


def test_panel_has_no_look_ahead() -> None:
    c = closes_panel()
    full = price_factor_frames(c)
    for cut in [300, 451]:
        truncated = price_factor_frames(c.iloc[: cut + 1])
        ts = c.index[cut]
        for name in full:
            a, b = full[name].loc[ts], truncated[name].loc[ts]
            pd.testing.assert_series_equal(a, b, check_names=False)


def test_ic_history_only_uses_closed_windows() -> None:
    c = closes_panel()
    panel = FactorPanel.from_closes(c, horizon=21)
    ts = c.index[500]
    hist = panel.ic_history(ts, periods=6)
    assert len(hist) == 6
    assert hist.index.max() == c.index[500 - 21]
    assert all(c.index.get_loc(d) + 21 <= 500 for d in hist.index)
    assert panel.ic_history(c.index[10], 6).empty


def test_daily_ic_detects_a_planted_signal() -> None:
    c = closes_panel()
    panel = FactorPanel.from_closes(c, horizon=21)
    panel.add("oracle", forward_returns_daily(c, 21))  # look-ahead on purpose
    ic = panel.daily_ic()
    assert ic["oracle"].dropna().mean() == pytest.approx(1.0)
    assert abs(ic["mom_12_1"].dropna().mean()) < 0.15  # random walk: no skill


def test_standardisation_steps() -> None:
    s = pd.Series(np.r_[np.arange(99.0), 1e6], index=[f"x{i}" for i in range(100)])
    assert winsorize(s).max() < 1e3
    z = standardize(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    sectors = {k: ("A" if i % 2 else "B") for i, k in enumerate(s.index)}
    sn = sector_neutral(z, sectors)
    assert sn.groupby(pd.Series(sectors)).mean().abs().max() < 1e-12


def test_composite_and_ic_weights() -> None:
    a = pd.Series({"x": 1.0, "y": -1.0, "z": 0.0})
    b = pd.Series({"x": 1.0, "y": 1.0})  # z missing
    comp = composite({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
    assert comp["x"] == 1.0 and comp["y"] == 0.0 and comp["z"] == 0.0
    assert "z" not in composite({"a": a, "b": b}, {"a": 1, "b": 1}, min_factors=2)
    hist = pd.DataFrame({"good": [0.05] * 10, "bad": [-0.04] * 10})
    w = ic_weights(hist)
    assert w["good"] == 1.0 and w["bad"] == 0.0  # negative IC is not flipped
    assert ic_weights(hist.head(3)) == {"good": 1.0, "bad": 1.0}  # too little history
    # when weighting zeroes out factors, the coverage requirement shrinks with them
    only_a = composite({"a": a, "b": b}, {"a": 1.0, "b": 0.0}, min_factors=2)
    assert set(only_a.index) == {"x", "y", "z"}


def ctx_at(c: pd.DataFrame, pos: int, eligible=None) -> StrategyContext:
    ts = c.index[pos]
    prices = {k: D(str(round(v, 6))) for k, v in c.loc[ts].dropna().items()}
    return StrategyContext(
        ts,
        "p",
        prices,
        c.loc[:ts],
        PositionBook("p", None),
        D(1_000_000),
        lambda _: D(1),
        0,
        eligible,
    )


def test_multifactor_picks_planted_winners_and_respects_universe() -> None:
    c = closes_panel()
    panel = FactorPanel.from_closes(c)
    star = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    star[["EQ:S10", "EQ:S11"]] = 1.0
    panel.add("star", star)
    strat = MultiFactor(
        panel,
        factors=("star",),
        weighting="equal",
        top_n=2,
        sector_neutral=False,
        sizing="equal",
        max_weight=1.0,
    )
    w = strat.on_bar(ctx_at(c, 500))
    assert set(w) == {"EQ:S10", "EQ:S11"}
    eligible = frozenset(c.columns) - {"EQ:S10"}
    w2 = strat.on_bar(ctx_at(c, 500, eligible))
    assert "EQ:S10" not in w2


def test_trend_filter_cuts_exposure_in_downtrend() -> None:
    c = closes_panel()
    panel = FactorPanel.from_closes(c)
    falling = pd.Series(np.linspace(200, 100, len(c)), index=c.index)
    on = MultiFactor(
        panel,
        top_n=10,
        trend_filter=True,
        benchmark=falling,
        sizing="equal",
        max_weight=1.0,
        sector_neutral=False,
    )
    off = MultiFactor(panel, top_n=10, sizing="equal", max_weight=1.0, sector_neutral=False)
    assert sum(on.on_bar(ctx_at(c, 500)).values()) == pytest.approx(0.5)
    assert sum(off.on_bar(ctx_at(c, 500)).values()) == pytest.approx(0.98)
    with pytest.raises(ValueError):
        MultiFactor(panel, trend_filter=True)
    with pytest.raises(ValueError):
        MultiFactor(panel, factors=("nope",))
