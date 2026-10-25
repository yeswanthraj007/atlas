import numpy as np
import pandas as pd
import pytest

from atlas.factors.fundamental import company_factor_series
from atlas.marketdata.sec import extract_facts, point_in_time, ttm_series


def fact(start, end, val, filed, form="10-Q", fp="Q1"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "fp": fp, "accn": filed}
    if start:
        d["start"] = start
    return d


def company(ni_rows, assets_rows=(), equity_rows=(), shares_rows=(), extra_tags=None):
    gaap = {
        "NetIncomeLoss": {"units": {"USD": list(ni_rows)}},
        "Assets": {"units": {"USD": list(assets_rows)}},
        "StockholdersEquity": {"units": {"USD": list(equity_rows)}},
    }
    gaap.update(extra_tags or {})
    return {
        "facts": {
            "us-gaap": gaap,
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": list(shares_rows)}}},
        }
    }


# Fiscal year = calendar year. FY2023 NI 400; 2024 Q1 YTD 120, H1 YTD 250; 2023 same YTDs.
NI = [
    fact("2023-01-01", "2023-12-31", 400, "2024-02-15", "10-K", "FY"),
    fact("2023-01-01", "2023-03-31", 100, "2023-05-01"),
    fact("2023-01-01", "2023-06-30", 210, "2023-08-01", fp="Q2"),
    fact("2024-01-01", "2024-03-31", 120, "2024-05-01"),
    fact("2024-01-01", "2024-06-30", 250, "2024-08-01", fp="Q2"),
]


def test_ttm_from_annual_plus_ytd() -> None:
    f = extract_facts(company(NI))
    ttm = ttm_series(f, "net_income").set_index("end")["val"]
    assert ttm[pd.Timestamp("2023-12-31")] == 400
    assert ttm[pd.Timestamp("2024-03-31")] == 400 + 120 - 100
    assert ttm[pd.Timestamp("2024-06-30")] == 400 + 250 - 210


def test_point_in_time_waits_for_filing_and_ignores_restatements() -> None:
    restated = [*NI, fact("2024-01-01", "2024-03-31", 999, "2024-09-01")]  # later restatement
    f = extract_facts(company(restated))
    s = point_in_time(
        ttm_series(f, "net_income"), pd.bdate_range("2024-04-25", "2024-09-10", tz="UTC")
    )
    assert s["2024-04-30"] == 400  # Q1 not filed yet
    assert s["2024-05-01"] == 400  # filed that day: usable only next business day
    assert s["2024-05-02"] == 420
    assert s["2024-08-02"] == 440
    assert s["2024-09-05"] == 440  # the restated Q1 (an older period) never overrides


def test_tag_priority_and_dual_class_shares() -> None:
    revenues = {
        "Revenues": {
            "units": {"USD": [fact("2023-01-01", "2023-12-31", 50, "2024-02-15", "10-K", "FY")]}
        },
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {"USD": [fact("2023-01-01", "2023-12-31", 60, "2024-02-15", "10-K", "FY")]}
        },
    }
    shares = [
        fact(None, "2024-01-31", 300, "2024-02-15"),
        fact(None, "2024-01-31", 200, "2024-02-15"),
    ]
    f = extract_facts(company(NI, shares_rows=shares, extra_tags=revenues))
    assert ttm_series(f, "revenue")["val"].tolist() == [60]  # preferred tag wins
    assert f[f["item"] == "shares"]["val"].tolist() == [500]  # class A + class B


def test_market_cap_survives_a_stock_split() -> None:
    """Shares reported before a 4-for-1 split must not make the stock look 4x cheaper."""
    dates = pd.bdate_range("2024-01-01", "2024-06-28", tz="UTC")
    split_day = pd.Timestamp("2024-04-01", tz="UTC")
    raw = pd.Series(np.where(dates < split_day, 400.0, 100.0), index=dates)
    split_adj = pd.Series(100.0, index=dates)  # history divided by 4 before the split
    f = extract_facts(
        company(
            NI,
            assets_rows=[fact(None, "2023-12-31", 1000, "2024-02-15", "10-K", "FY")],
            equity_rows=[fact(None, "2023-12-31", 500, "2024-02-15", "10-K", "FY")],
            shares_rows=[fact(None, "2024-01-31", 10, "2024-02-15")],  # pre-split count
        )
    )
    out = company_factor_series(f, split_adj, raw, dates)
    mcap = out["market_cap"].dropna()
    assert mcap.iloc[0] == pytest.approx(10 * 400)  # before the split
    assert mcap.loc["2024-05-01"] == pytest.approx(10 * 4 * 100)  # same company value after
    assert out["book_to_price"].loc["2024-05-01"] == pytest.approx(500 / 4000)
    assert out["earnings_yield"].loc["2024-05-01"] == pytest.approx(400 / 4000)
    assert np.isnan(out["earnings_yield"].loc["2024-02-15"])  # filed that day: not yet usable
