from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from atlas.cli import main
from atlas.config import MissingCredentialsError, Settings
from atlas.marketdata.alpaca import AlpacaBarProvider
from atlas.marketdata.bars import Adjustment, Timeframe, normalize
from atlas.marketdata.fama_french import parse_french_csv
from atlas.marketdata.loader import load_for_backtest, returns_panel
from atlas.marketdata.quality import Severity, check_bars, has_errors
from atlas.marketdata.store import BarStore
from atlas.refdata.instruments import SecurityMaster


def bars(n=30, start="2026-01-02", seed=0, tz=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, tz=tz)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )


def test_normalize_makes_utc_sorted_unique() -> None:
    df = bars(5)
    shuffled = pd.concat([df.iloc[[3, 1, 0, 2, 4]], df.iloc[[1]]])
    out = normalize(shuffled)
    assert str(out.index.tz) == "UTC" and out.index.is_monotonic_increasing
    assert out.index.is_unique and len(out) == 5
    with pytest.raises(ValueError):
        normalize(df.drop(columns="volume"))


def test_store_roundtrip_merge_and_replace(tmp_path) -> None:
    store = BarStore(tmp_path)
    df = bars(30)
    store.write("AAPL", df.iloc[:20], adjustment=Adjustment.RAW)
    store.write("AAPL", df.iloc[15:], adjustment=Adjustment.RAW)  # overlapping append
    raw = store.read("AAPL", adjustment=Adjustment.RAW)
    assert len(raw) == 30
    store.write("AAPL", df.iloc[:20], adjustment=Adjustment.ALL)
    store.write("AAPL", df.iloc[20:], adjustment=Adjustment.ALL)  # adjusted: replaced whole
    assert len(store.read("AAPL")) == 10
    assert store.symbols(adjustment=Adjustment.RAW) == ["AAPL"]
    window = store.read("AAPL", adjustment=Adjustment.RAW, start=datetime(2026, 1, 10, tzinfo=UTC))
    assert window.index[0] >= pd.Timestamp("2026-01-10", tz="UTC")
    with pytest.raises(FileNotFoundError):
        store.read("MSFT")
    assert not list(tmp_path.rglob("*.tmp"))


def test_quality_checks_flag_each_problem() -> None:
    df = normalize(bars(40))
    assert check_bars("OK", df) == []
    bad = df.copy()
    bad.iloc[3, bad.columns.get_loc("close")] = -1.0
    bad.iloc[5, bad.columns.get_loc("high")] = bad["low"].iloc[5] * 0.5
    bad.iloc[8, bad.columns.get_loc("open")] = np.nan
    bad.iloc[20:27, bad.columns.get_loc("close")] = bad["close"].iloc[20]
    bad.iloc[30, bad.columns.get_loc("close")] = bad["close"].iloc[29] * 2.0
    found = {i.check: i for i in check_bars("BAD", bad)}
    for check in [
        "non_positive_price",
        "ohlc_inconsistent",
        "nan",
        "stale_price",
        "extreme_return",
    ]:
        assert check in found, check
    assert found["nan"].severity is Severity.ERROR
    assert found["extreme_return"].severity is Severity.WARN
    assert has_errors(list(found.values()))
    gappy = df.drop(df.index[10:20])
    assert any(i.check == "gap" for i in check_bars("GAP", gappy))


FRENCH_SAMPLE = (
    "This file was created by using the 202608 CRSP database.\r\n"
    "Missing data are indicated by -99.99 or -999.\r\n\r\n"
    ",Mkt-RF,SMB,HML,RMW,CMA,RF\r\n"
    "20260827,    0.50,   -0.20,    0.10,   -99.99,    0.00,    0.01\r\n"
    "20260828,   -0.34,   -0.37,    0.28,    1.66,    0.49,    0.01\r\n"
    "\r\n Annual Factors: January-December \r\n"
    ",Mkt-RF,SMB,HML,RMW,CMA,RF\r\n"
    "2025,   10.00,    1.00,    1.00,    1.00,    1.00,    4.00\r\n"
    "\r\nCopyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
)


def test_parse_french_csv() -> None:
    df = parse_french_csv(FRENCH_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 2  # stops at the end of the daily table
    assert df.loc["2026-08-28", "Mkt-RF"].item() == pytest.approx(-0.0034)
    assert np.isnan(df.loc["2026-08-27", "RMW"].item())
    with pytest.raises(ValueError):
        parse_french_csv("no table here")


class FakeAlpacaClient:
    def __init__(self) -> None:
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        frames = []
        for sym in request.symbol_or_symbols:
            df = bars(5, tz="UTC")
            df["symbol"] = sym
            frames.append(df.rename_axis("timestamp").reset_index())
        multi = pd.concat(frames).set_index(["symbol", "timestamp"])
        return SimpleNamespace(df=multi)


def settings(**kw) -> Settings:
    base = dict(
        alpaca_key="k", alpaca_secret="s", alpaca_paper=True, alpaca_feed="iex", data_dir="data"
    )
    base.update(kw)
    return Settings(**base)


def test_alpaca_provider_batches_and_normalizes() -> None:
    client = FakeAlpacaClient()
    provider = AlpacaBarProvider(settings(), client=client, batch_size=2)
    out = provider.get_bars(
        ["MSFT", "AAPL", "SPY", "AAPL"],
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 2, 1, tzinfo=UTC),
        Timeframe.DAY,
        Adjustment.ALL,
    )
    assert sorted(out) == ["AAPL", "MSFT", "SPY"]
    assert len(client.requests) == 2  # 3 unique symbols, batches of 2
    assert client.requests[0].adjustment.value == "all"
    assert client.requests[0].feed.value == "iex"
    assert str(out["AAPL"].index.tz) == "UTC"


def test_missing_credentials_are_explicit(monkeypatch, tmp_path) -> None:
    with pytest.raises(MissingCredentialsError):
        settings(alpaca_key=None).require_alpaca()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert main(["data", "download", "AAPL"]) == 2


def test_loader_refuses_broken_data(tmp_path) -> None:
    store = BarStore(tmp_path)
    store.write("GOOD", bars(30, seed=1))
    broken = bars(30, seed=2)
    broken.iloc[4, 3] = -5.0
    store.write("BROKEN", broken)
    sm = SecurityMaster()
    out = load_for_backtest(store, ["GOOD"], sm)
    assert list(out) == ["EQ:GOOD"] and "EQ:GOOD" in sm
    with pytest.raises(ValueError, match="BROKEN"):
        load_for_backtest(store, ["GOOD", "BROKEN"], SecurityMaster())
    lenient = load_for_backtest(store, ["GOOD", "BROKEN"], SecurityMaster(), strict=False)
    assert len(lenient) == 2
    assert returns_panel(out).shape == (29, 1)
