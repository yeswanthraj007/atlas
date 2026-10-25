import numpy as np
import pandas as pd
import pytest

from atlas.factors.earnings import company_earnings_series
from atlas.marketdata.sec import extract_facts
from atlas.ml.volatility import build_vol_dataset, forward_log_vol, log_realized_vol
from tests.test_fundamentals import NI, company, fact


def test_earnings_surprise_and_filing_clock_are_point_in_time() -> None:
    assets = [fact(None, "2023-12-31", 1000, "2024-02-15", "10-K", "FY")]
    f = extract_facts(company(NI, assets_rows=assets))
    dates = pd.bdate_range("2024-04-25", "2024-08-30", tz="UTC")
    out = company_earnings_series(f, dates)
    es = out["earnings_surprise"]
    # Q1-2024 filed 2024-05-01: TTM 400 -> 420, so surprise = +20 / 1000 assets
    assert np.isnan(es["2024-05-01"]) or es["2024-05-01"] != pytest.approx(0.02)
    assert es["2024-05-02"] == pytest.approx(0.02)
    assert es["2024-08-02"] == pytest.approx(0.02)  # Q2: 420 -> 440, again +20
    days = out["days_since_filing"]
    assert days["2024-05-02"] == 0 and days["2024-05-09"] == 5


def test_forward_volatility_uses_only_future_returns() -> None:
    idx = pd.bdate_range("2024-01-01", periods=120, tz="UTC")
    rng = np.random.default_rng(1)
    r = np.r_[rng.normal(0, 0.01, 60), rng.normal(0, 0.04, 60)]  # vol jumps at bar 60
    closes = pd.DataFrame({"EQ:A": 100 * np.exp(np.cumsum(r))}, index=idx)
    fwd = np.exp(forward_log_vol(closes, 21))["EQ:A"]
    back = np.exp(log_realized_vol(closes, 21))["EQ:A"]
    assert fwd.iloc[38] < 0.3 and fwd.iloc[60] > 0.45  # at 60 the window is all high-vol
    assert back.iloc[60] < 0.3  # looking backwards at 60 still sees calm
    # changing prices after t must not change features at t
    altered = closes.copy()
    altered.iloc[80:] *= 3
    a = log_realized_vol(altered, 21)["EQ:A"].iloc[79]
    assert a == pytest.approx(log_realized_vol(closes, 21)["EQ:A"].iloc[79])
    data = build_vol_dataset(closes, idx[[30, 70]], 21)
    assert list(data.index.get_level_values("date")) == [idx[70]]  # day 30 lacks 63d history
    assert data["label_end"].iloc[0] == idx[91]
