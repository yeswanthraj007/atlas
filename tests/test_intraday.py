import numpy as np
import pandas as pd

from atlas.factors.intraday import daily_pnl, session_table, summarize


def _session(day: str, base: float, early_close: bool = False) -> pd.DataFrame:
    """Minute bars 09:30..15:59 ET: price = base + minute index x 0.01, open = close - 0.005."""
    idx = pd.date_range(f"{day} 09:30", f"{day} {'12:59' if early_close else '15:59'}", freq="min",
                        tz="America/New_York").tz_convert("UTC")  # fmt: skip
    close = base + np.arange(len(idx)) * 0.01
    return pd.DataFrame({"open": close - 0.005, "high": close, "low": close - 0.01, "close": close,
                         "volume": 1000.0}, index=idx)  # fmt: skip


def test_session_table_uses_the_right_bars_and_drops_incomplete_days():
    bars = pd.concat([_session("2024-03-04", 100.0), _session("2024-03-05", 101.0),
                      _session("2024-03-06", 102.0, early_close=True)])  # fmt: skip
    t = session_table(bars)
    assert list(t.index) == [
        pd.Timestamp("2024-03-05")
    ]  # 03-04 has no previous close; 03-06 is an early close
    d = pd.Timestamp("2024-03-05")
    row = t.loc[d]
    close_prev = 100.0 + 389 * 0.01  # 15:59 bar of the previous day
    assert np.isclose(row["p1000"], 101.0 + 29 * 0.01)  # close of the 09:59 bar is minute 29
    assert np.isclose(row["r1"], np.log(row["p1000"] / close_prev))
    assert np.isclose(row["p1500"], 101.0 + 329 * 0.01) and np.isclose(
        row["p1530"], 101.0 + 359 * 0.01
    )
    assert np.isclose(row["entry"], 101.0 + 360 * 0.01 - 0.005)  # OPEN of the 15:30 bar
    assert np.isclose(row["trade_ret"], np.log((101.0 + 389 * 0.01) / (101.0 + 360 * 0.01 - 0.005)))
    assert pd.Timestamp("2024-03-06") not in t.index  # early close: no 15:30 bars


def test_nothing_after_the_decision_time_enters_the_signal():
    days = pd.bdate_range("2024-01-02", periods=40)
    bars = pd.concat([_session(str(d.date()), 100 + i) for i, d in enumerate(days)])
    a = session_table(bars)
    b = bars.copy()
    et = b.index.tz_convert("America/New_York")
    last_day = et.normalize().tz_localize(None) == days[-1]
    late = (et.hour * 60 + et.minute) >= 931  # everything after the entry bar
    b.loc[last_day & late, ["open", "high", "low", "close"]] *= (
        1.05  # only the final day, after the 15:30 entry bar
    )
    c = session_table(b)
    cols = ["r1", "r12", "vol20"]
    # every earlier day is untouched, and so are the final day's signals; only its outcome moves
    pd.testing.assert_frame_equal(a[cols], c[cols])
    assert not np.isclose(a["trade_ret"].iloc[-1], c["trade_ret"].iloc[-1])
    assert np.allclose(a["trade_ret"].iloc[:-1], c["trade_ret"].iloc[:-1])


def test_pnl_costs_and_summary():
    pos = pd.Series([1, -1, 0, 1])
    ret = pd.Series([0.001, -0.002, 0.003, -0.0005])
    pnl = daily_pnl(pos, ret, cost_bp=1.0)
    assert np.allclose(pnl, [0.001 - 1e-4, 0.002 - 1e-4, 0.0, -0.0005 - 1e-4])
    s = summarize(pnl, pos)
    assert s["trades"] == 3 and s["days"] == 4 and abs(s["hit_rate"] - 2 / 3) < 1e-9
    assert abs(s["mean_net_bp"] - np.mean([9.0, 19.0, -6.0])) < 1e-9
