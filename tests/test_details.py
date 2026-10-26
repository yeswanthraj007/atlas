import pandas as pd

from atlas.recommend.details import _p_abs_above, _ret


def test_move_probability_matches_normal_table():
    # P(|Z| > 1) = 0.3173, P(|Z| > 2) = 0.0455 for a standard normal
    assert abs(_p_abs_above(0.10, 0.10) - 0.3173) < 1e-3
    assert abs(_p_abs_above(0.20, 0.10) - 0.0455) < 1e-3
    assert _p_abs_above(0.10, 0.0) == 0.0


def test_returns_use_the_right_window():
    c = pd.Series([100.0, 110.0, 121.0, 133.1])
    assert abs(_ret(c, 3) - 0.331) < 1e-9  # 100 -> 133.1
    assert abs(_ret(c, 3, skip=1) - 0.21) < 1e-9  # 100 -> 121, skipping the last day
    assert _ret(c, 10) is None


def test_earnings_calendar_uses_announcement_gaps(tmp_path):
    from atlas.recommend.details import earnings_calendar

    times = ["2026-02-01", "2026-05-01", "2026-08-01", "2026-12-01"]  # last one is in the future
    pd.DataFrame({"symbol": "X", "accepted": [f"{t}T21:00:00Z" for t in times]}).to_parquet(
        tmp_path / "e.parquet"
    )
    cal = earnings_calendar(pd.Timestamp("2026-10-26", tz="UTC"), tmp_path / "e.parquet")
    since, nxt = cal["X"]
    assert (
        since == 85
    )  # Aug 1 21:00 UTC -> Oct 26 00:00 UTC is 85 full days; future release ignored
    assert nxt == 90.5 - 85  # median of the gaps Feb->May (89 days) and May->Aug (92 days)


def test_report_json_has_no_nan():
    import json
    import math

    from atlas.recommend.daily import _no_nan

    out = _no_nan({"a": math.nan, "b": [1.0, math.inf, {"c": float("nan")}], "d": "x"})
    assert out == {"a": None, "b": [1.0, None, {"c": None}], "d": "x"}
    json.dumps(out, allow_nan=False)
