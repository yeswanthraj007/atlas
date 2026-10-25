from datetime import date

import numpy as np
import pandas as pd

from atlas.recommend.engine import Registry, SignalRecord, build_watchlist


def registry(news_t: float) -> Registry:
    r = Registry()
    r.record(SignalRecord("news_sentiment", "news", "direction", "ic", 0.02, news_t, "2025-26"))
    r.record(SignalRecord("vol_forecast", "vol", "risk", "acc_above_median", 0.76, 12.0, "2025-26"))
    return r


SIGNALS = {"news_sentiment": pd.Series({"AAA": 0.4, "BBB": -0.3, "CCC": 0.1})}
VOL = pd.Series({"AAA": 0.2, "BBB": 0.6, "CCC": 0.35})


def test_unvalidated_signal_never_ranks() -> None:
    wl = build_watchlist(date(2026, 10, 23), SIGNALS, VOL, None, registry(news_t=1.1))
    assert wl.picks.empty and wl.used_signals == []
    assert "news_sentiment" in wl.context_signals
    assert list(wl.risk_alerts.index[:1]) == ["BBB"]  # risk ranking is validated
    md = wl.to_markdown()
    assert "No validated direction signal" in md and "not advice" in md


def test_validated_signal_ranks_and_registry_roundtrips(tmp_path) -> None:
    reg = registry(news_t=3.0)
    wl = build_watchlist(
        date(2026, 10, 23), SIGNALS, VOL, {"days_since_filing": pd.Series({"AAA": 60})}, reg
    )
    assert list(wl.picks.index) == ["AAA", "CCC", "BBB"]
    assert wl.picks.loc["AAA", "days_since_filing"] == 60
    path = tmp_path / "reg.json"
    reg.save(path)
    again = Registry.load(path)
    assert again.validated() == ["news_sentiment", "vol_forecast"]
    assert np.isnan(
        Registry.load(tmp_path / "missing.json")
        .signals.get("x", SignalRecord("x", "", "direction"))
        .holdout_t
    )
