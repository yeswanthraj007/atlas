import numpy as np
import pandas as pd
import pytest

from atlas.scout import Call, ScoutReport, Theme, load_reports, save_report, score_reports


def report(rid="2026-10-25", through="2026-01-05", horizon=5, direction="outperform"):
    return ScoutReport(
        rid,
        "2026-10-25T12:00:00+00:00",
        through,
        "claude",
        [
            Theme(
                "AI memory demand",
                ["Information Technology"],
                "Data-centre build-out lifts memory",
                ["headline"],
                [
                    Call(
                        "AAA",
                        direction,
                        horizon,
                        "customer link: hyperscaler orders",
                        "customer",
                        4,
                    )
                ],
            )
        ],
    )


def test_reports_are_validated_and_append_only(tmp_path) -> None:
    save_report(report(), tmp_path)
    with pytest.raises(FileExistsError):
        save_report(report(), tmp_path)
    bad = report(rid="x")
    bad.themes[0].sectors = ["Crypto"]
    with pytest.raises(ValueError, match="unknown sectors"):
        save_report(bad, tmp_path)
    assert load_reports(tmp_path)[0].themes[0].calls[0].ticker == "AAA"


def test_scoring_uses_only_prices_after_the_report() -> None:
    idx = pd.bdate_range("2026-01-01", periods=30, tz="UTC")
    closes = pd.DataFrame({"EQ:AAA": np.linspace(100, 130, 30)}, index=idx)
    spy = pd.Series(100.0, index=idx)
    s = score_reports([report(through="2026-01-05", horizon=5)], closes, spy)
    start = idx.searchsorted(pd.Timestamp("2026-01-05", tz="UTC"), side="right")
    expected = closes["EQ:AAA"].iloc[start + 5] / closes["EQ:AAA"].iloc[start] - 1
    assert s.loc[0, "status"] == "scored" and s.loc[0, "hit"]
    assert s.loc[0, "excess"] == pytest.approx(expected)
    pending = score_reports([report(through="2026-02-09", horizon=20)], closes, spy)
    assert pending.loc[0, "status"] == "pending"
