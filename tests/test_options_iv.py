from datetime import date
from types import SimpleNamespace as NS

import pandas as pd

from atlas.marketdata.options_iv import (
    OptionIVRecorder,
    load_iv,
    parse_occ,
    summarize_chain,
)

ASOF = date(2026, 11, 3)


def snap(bid, ask, iv, delta=None):
    return NS(
        latest_quote=NS(bid_price=bid, ask_price=ask),
        implied_volatility=iv,
        greeks=NS(delta=delta) if delta is not None else None,
    )


def test_parse_occ():
    c = parse_occ("AAPL261116C00360000")
    assert (c.root, c.expiry, c.kind, c.strike) == ("AAPL", date(2026, 11, 16), "C", 360.0)
    assert parse_occ("not-a-contract") is None


def test_summary_uses_the_atm_strike_and_computes_straddle():
    chain = {
        "XYZ261116C00100000": snap(4.9, 5.1, 0.40, 0.52),
        "XYZ261116P00100000": snap(4.4, 4.6, 0.44, -0.48),
        "XYZ261116C00090000": snap(11.0, 11.4, 0.45, 0.80),  # not ATM
        "XYZ261116P00090000": snap(0.4, 0.5, 0.52, -0.25),
        "XYZ261116C00110000": snap(1.0, 1.1, 0.38, 0.25),
        "XYZ261116P00110000": snap(10.0, 10.5, 0.46, -0.77),
    }
    rows = summarize_chain(chain, spot=100.2, asof=ASOF)
    assert len(rows) == 1
    r = rows[0]
    assert r["atm_strike"] == 100.0 and r["dte"] == 13
    assert abs(r["atm_iv"] - 0.42) < 1e-9
    assert abs(r["straddle_pct"] - (5.0 + 4.5) / 100.2) < 1e-9
    assert abs(r["skew25"] - (0.52 - 0.38)) < 1e-9  # 25-delta put IV minus call IV
    assert abs(r["straddle_tv_pct"] - ((5.0 + 4.5) / 100.2 - 0.2 / 100.2)) < 1e-9


def test_bad_quotes_and_ivs_are_dropped_not_used():
    chain = {
        "XYZ261116C00100000": snap(0.0, 5.0, 9.0, 0.5),  # zero bid, absurd IV
        "XYZ261116P00100000": snap(4.4, 4.6, 0.44, -0.5),
        "XYZ270301C00100000": snap(5.0, 5.2, 0.4, 0.5),  # beyond 70 days
        "XYZ270301P00100000": snap(5.0, 5.2, 0.4, -0.5),
    }
    rows = summarize_chain(chain, spot=100.0, asof=ASOF)
    assert len(rows) == 1
    r = rows[0]
    assert r["call_iv"] != r["call_iv"]  # NaN
    assert r["atm_iv"] == 0.44 and r["straddle_pct"] != r["straddle_pct"]


class FakeOptions:
    def __init__(self, chains):
        self.chains, self.calls = chains, []

    def get_option_chain(self, req):
        self.calls.append(req.underlying_symbol)
        return self.chains.get(req.underlying_symbol, {})


class FakeStocks:
    def get_stock_snapshot(self, req):
        ts = pd.Timestamp("2026-11-02 21:00", tz="UTC")
        return {
            s: NS(latest_trade=NS(price=100.0, timestamp=ts), daily_bar=None)
            for s in req.symbol_or_symbols
            if s != "NOPRICE"
        }


def test_recorder_writes_resumes_and_marks_symbols_without_options(tmp_path):
    chain = {
        "AAA261116C00100000": snap(4.9, 5.1, 0.4, 0.5),
        "AAA261116P00100000": snap(4.4, 4.6, 0.4, -0.5),
    }
    opts = FakeOptions({"AAA": chain})
    rec = OptionIVRecorder(opts, FakeStocks(), min_interval=0, root=tmp_path, sleep=lambda _: None)
    path, failed = rec.record(["AAA", "BBB", "NOPRICE"], progress=lambda _: None)
    assert failed == ["NOPRICE"] and path.name == "2026-11-02.parquet"
    iv = load_iv(tmp_path)
    assert list(iv["symbol"]) == ["AAA"]  # BBB has no options: marked, not returned
    raw = pd.read_parquet(path)
    assert set(raw["symbol"]) == {"AAA", "BBB"}
    rec.record(["AAA", "BBB"], progress=lambda _: None)  # resume: nothing re-requested
    assert opts.calls == ["AAA", "BBB"]
