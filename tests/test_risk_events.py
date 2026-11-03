import numpy as np
import pandas as pd

from atlas.ml.risk_events import (
    build_drawdown_samples,
    build_events,
    daily_features,
    days_since_event_panel,
    paired_t,
    predict_above_median,
    wiki_features,
)


def _prices(n=400, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01 05:00", periods=n, freq="B", tz="UTC")
    c = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0)),
        index=idx, columns=[f"EQ:S{i}" for i in range(k)],
    )  # fmt: skip
    vol = pd.DataFrame(
        rng.integers(1_000, 9_000, (n, k)).astype(float), index=idx, columns=c.columns
    )
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return c, vol, spy


def test_event_reaction_window_after_hours_vs_before_open():
    c, _, spy = _prices()
    day = c.index[100]  # a trading day
    # accepted 21:00 UTC on `day` (after the close) -> reaction is the NEXT session
    after = pd.Timestamp(day.date()).tz_localize("UTC") + pd.Timedelta(hours=21)
    # accepted 11:00 UTC on c.index[200] (before the open) -> reaction is that same day
    before = pd.Timestamp(c.index[200].date()).tz_localize("UTC") + pd.Timedelta(hours=11)
    e = build_events(
        c, spy,
        pd.DataFrame({"symbol": ["S0", "S0"], "accepted": [after, before], "items": "2.02"}),
    )  # fmt: skip
    assert list(e["d0"]) == [101, 200]
    r = c["EQ:S0"].to_numpy()
    s = spy.to_numpy()
    assert np.isclose(e["react"].iloc[0], (r[101] / r[100] - 1) - (s[101] / s[100] - 1))
    assert np.isclose(e["react"].iloc[1], (r[200] / r[199] - 1) - (s[200] / s[199] - 1))
    assert list(e["p"]) == [100, 199]  # features use the last close BEFORE the window


def test_event_history_and_targets_use_only_the_past():
    c, _, spy = _prices(n=900, k=1)
    times = [pd.Timestamp(c.index[40 + 60 * i].date()).tz_localize("UTC") + pd.Timedelta(hours=21)
             for i in range(12)]  # fmt: skip
    ev = pd.DataFrame({"symbol": "S0", "accepted": times, "items": "2.02"})
    e = build_events(c, spy, ev)
    a = e["abs_react"].to_numpy()
    assert np.isnan(e["prev1"].iloc[0]) and np.isclose(e["prev1"].iloc[5], a[4])
    assert np.isclose(e["med8"].iloc[9], np.median(a[1:9]))  # previous 8, excluding itself
    assert e["y_e2"].iloc[:3].isna().all()  # fewer than 4 prior events: unknown
    assert e["y_e2"].iloc[9] == float(a[9] > np.median(a[1:9]))
    # changing a LATER event must not change this event's history features
    ev2 = ev.copy()
    c2 = c.copy()
    c2.iloc[40 + 60 * 11 + 1 :] *= 3.0
    e2 = build_events(c2, spy, ev2)
    cols = ["prev1", "prev2_mean", "prev4_mean", "med8"]
    pd.testing.assert_frame_equal(e[cols].iloc[:11], e2[cols].iloc[:11])


def test_days_since_event_counts_trading_days():
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    d0 = pd.DataFrame({"instrument": ["EQ:A", "EQ:A"], "d0": [2, 7]})
    out = days_since_event_panel(idx, pd.Index(["EQ:A", "EQ:B"]), d0)
    assert np.isnan(out["EQ:A"].iloc[:2]).all()
    assert list(out["EQ:A"].iloc[2:10]) == [0, 1, 2, 3, 4, 0, 1, 2]
    assert out["EQ:B"].isna().all()


def test_drawdown_target_and_no_lookahead_in_features():
    c, vol, spy = _prices(n=500, k=40)
    mem = pd.DataFrame(True, index=c.index, columns=c.columns)
    s = build_drawdown_samples(c, mem)
    row = s.iloc[0]
    p, j = int(row["p"]), int(row["stock"])
    expect = c.iloc[p + 1 : p + 22, j].min() / c.iloc[p, j] - 1
    assert np.isclose(row["fwd_dd"], expect)
    first = s[s["p"] == s["p"].min()]
    assert abs(first["y_d1"].mean() - 0.5) < 0.05  # balanced within a date
    dse = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    f1 = daily_features(c, vol, c, spy, dse)
    c2, vol2 = c.copy(), vol.copy()
    c2.iloc[300:] *= 2.0
    vol2.iloc[300:] *= 5.0
    f2 = daily_features(c2, vol2, c2, spy, dse)
    for k in f1:
        if k != "days_since_event":
            np.testing.assert_allclose(f1[k][:300], f2[k][:300], equal_nan=True, err_msg=k)


def test_wiki_features_are_lagged_one_day():
    idx = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
    days = pd.date_range("2019-06-01", periods=500, freq="D", tz="UTC")
    views = pd.DataFrame({"S0": np.full(len(days), 100.0)}, index=days)
    spike_day = idx[150]
    views.loc[spike_day.normalize()] = 1_000_000.0
    f = wiki_features(views, idx, pd.Index(["EQ:S0"]))
    assert abs(f["wiki_chg"][150, 0]) < 1e-6  # the day itself is not visible yet
    assert f["wiki_chg"][151, 0] > 1.0  # visible one day later


def test_predict_above_median_and_paired_t():
    s = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    g = np.array([0, 0, 0, 1, 1, 1])
    assert list(predict_above_median(s, g)) == [0, 0, 1, 0, 0, 1]
    a = pd.Series([0.7, 0.72, 0.71, 0.69])
    assert paired_t(a, a - 0.05) > 100 or np.isinf(paired_t(a, a - 0.05))
