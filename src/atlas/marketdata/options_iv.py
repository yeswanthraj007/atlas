"""Daily recorder of option-implied volatility, saved as compact per-expiry summaries.

Why: the options market's own forecast of how much a stock will move (the ATM straddle
before earnings, the term structure of implied volatility) is information our price-based
models do not have, and Alpaca only serves the current chain, with no history. So we record
it every day. In six to twelve months it becomes a tested feature (for example for the
earnings-move size model, trial E2).

For each underlying and each expiry within MAX_DTE days we keep, from the at-the-money
strike: call and put implied volatility (Alpaca's, from the indicative feed), the
straddle price as a share of the stock price, its bid-ask spread, and a 25-delta skew.
Raw chains (about a million contracts a day) are not stored.

Files: data/options_iv/<as-of date>.parquet, one row per (symbol, expiry), merged by key if
the job is re-run on the same day. Resumable: symbols already in today's file are skipped.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("data/options_iv")
MAX_DTE = 70
BAND = 0.12  # strikes within +-12% of the stock price
IV_RANGE = (0.03, 5.0)
MAX_SPREAD = 0.5  # drop quotes wider than 50% of their midpoint

_OCC = re.compile(r"^(?P<root>[A-Z]{1,6}\d?)(?P<exp>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class Contract:
    root: str
    expiry: date
    kind: str  # "C" or "P"
    strike: float


def parse_occ(symbol: str) -> Contract | None:
    m = _OCC.match(symbol)
    if not m:
        return None
    e = m["exp"]
    return Contract(
        m["root"],
        date(2000 + int(e[:2]), int(e[2:4]), int(e[4:6])),
        m["cp"],
        int(m["strike"]) / 1000.0,
    )


def _mid(snap: Any) -> tuple[float, float] | None:
    """(midpoint, relative spread) from a snapshot's latest quote, or None if unusable."""
    q = getattr(snap, "latest_quote", None)
    if q is None or q.bid_price is None or q.ask_price is None:
        return None
    bid, ask = float(q.bid_price), float(q.ask_price)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / mid
    return (mid, spread) if spread <= MAX_SPREAD else None


def _iv(snap: Any) -> float:
    v = getattr(snap, "implied_volatility", None)
    if v is None or not (IV_RANGE[0] <= float(v) <= IV_RANGE[1]):
        return float("nan")
    return float(v)


def _delta(snap: Any) -> float:
    g = getattr(snap, "greeks", None)
    d = getattr(g, "delta", None) if g is not None else None
    return float(d) if d is not None else float("nan")


def summarize_chain(chain: dict[str, Any], spot: float, asof: date) -> list[dict]:
    """One row per expiry from a chain snapshot dict {occ_symbol: snapshot}."""
    by_exp: dict[date, dict[float, dict[str, Any]]] = {}
    n_contracts: dict[date, int] = {}
    for sym, snap in chain.items():
        c = parse_occ(sym)
        if c is None:
            continue
        dte = (c.expiry - asof).days
        if not 1 <= dte <= MAX_DTE:
            continue
        by_exp.setdefault(c.expiry, {}).setdefault(c.strike, {})[c.kind] = snap
        n_contracts[c.expiry] = n_contracts.get(c.expiry, 0) + 1
    rows = []
    for exp, strikes in sorted(by_exp.items()):
        both = [k for k, v in strikes.items() if "C" in v and "P" in v]
        if not both:
            continue
        k = min(both, key=lambda s: abs(s - spot))
        call, put = strikes[k]["C"], strikes[k]["P"]
        civ, piv = _iv(call), _iv(put)
        ivs = [v for v in (civ, piv) if np.isfinite(v)]
        cm, pm = _mid(call), _mid(put)
        straddle = (cm[0] + pm[0]) / spot if cm and pm else float("nan")
        spread = (
            ((cm[0] * cm[1]) + (pm[0] * pm[1])) / (cm[0] + pm[0]) if cm and pm else float("nan")
        )
        puts25 = [(abs(_delta(v["P"]) + 0.25), _iv(v["P"])) for v in strikes.values() if "P" in v]
        calls25 = [(abs(_delta(v["C"]) - 0.25), _iv(v["C"])) for v in strikes.values() if "C" in v]
        puts25 = [x for x in puts25 if np.isfinite(x[0]) and np.isfinite(x[1]) and x[0] < 0.1]
        calls25 = [x for x in calls25 if np.isfinite(x[0]) and np.isfinite(x[1]) and x[0] < 0.1]
        skew = (
            min(puts25)[1] - min(calls25)[1] if puts25 and calls25 else float("nan")
        )  # 25-delta put IV minus 25-delta call IV
        rows.append(
            {
                "expiry": pd.Timestamp(exp),
                "dte": (exp - asof).days,
                "spot": spot,
                "atm_strike": k,
                "call_iv": civ,
                "put_iv": piv,
                "atm_iv": float(np.mean(ivs)) if ivs else float("nan"),
                "straddle_pct": straddle,
                # time value only (about twice an ATM option's time value): drops the
                # intrinsic part when the nearest strike is not exactly at the money
                "straddle_tv_pct": straddle - abs(spot - k) / spot,
                "straddle_spread_pct": spread,
                "skew25": skew,
                "n_contracts": n_contracts[exp],
            }
        )
    return rows


class OptionIVRecorder:
    def __init__(
        self,
        option_client: Any,
        stock_client: Any,
        min_interval: float = 0.35,  # free plan: 200 requests per minute
        root: str | Path = ROOT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.options, self.stocks = option_client, stock_client
        self.min_interval = min_interval
        self.root = Path(root)
        self._sleep = sleep
        self._last = 0.0

    @classmethod
    def from_settings(cls, settings: Any | None = None) -> OptionIVRecorder:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient

        from atlas.config import load_settings

        key, secret = (settings or load_settings()).require_alpaca()
        return cls(OptionHistoricalDataClient(key, secret), StockHistoricalDataClient(key, secret))

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()

    def spots(self, symbols: list[str]) -> dict[str, tuple[float, date]]:
        """Latest trade price and its date for each symbol (IEX feed; daily bar fallback)."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockSnapshotRequest

        out: dict[str, tuple[float, date]] = {}
        for i in range(0, len(symbols), 100):
            self._throttle()
            snaps = self.stocks.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=symbols[i : i + 100], feed=DataFeed.IEX)
            )
            for s, snap in snaps.items():
                t = getattr(snap, "latest_trade", None)
                bar = getattr(snap, "daily_bar", None)
                if t is not None and t.price:
                    out[s] = (
                        float(t.price),
                        pd.Timestamp(t.timestamp).tz_convert("America/New_York").date(),
                    )
                elif bar is not None and bar.close:
                    out[s] = (
                        float(bar.close),
                        pd.Timestamp(bar.timestamp).tz_convert("America/New_York").date(),
                    )
        return out

    def chain(self, symbol: str, spot: float, today: date) -> dict[str, Any]:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionChainRequest

        req = OptionChainRequest(
            underlying_symbol=symbol,
            feed=OptionsFeed.INDICATIVE,
            expiration_date_gte=today,
            expiration_date_lte=today + timedelta(days=MAX_DTE),
            strike_price_gte=round(spot * (1 - BAND), 2),
            strike_price_lte=round(spot * (1 + BAND), 2),
        )
        for attempt in range(4):
            self._throttle()
            try:
                return self.options.get_option_chain(req)
            except Exception as exc:  # rate limit or transient error: back off and retry
                if "no options" in str(exc).lower() or "not found" in str(exc).lower():
                    return {}
                self._sleep(3 * 2**attempt)
        raise RuntimeError(f"option chain request failed repeatedly for {symbol}")

    def record(
        self,
        symbols: Iterable[str],
        progress: Callable[[str], None] = print,
        aliases: dict[str, str] | None = None,
    ) -> tuple[Path | None, list[str]]:
        """`symbols` are option roots (what the chain request needs). `aliases` maps a root to
        the stock's own ticker where they differ (option root BRKB, stock BRK.B): the stock price
        is looked up under the ticker, and rows are saved under the ticker."""
        symbols = list(symbols)
        alias = aliases or {}
        raw = self.spots([alias.get(s, s) for s in symbols])
        spots = {s: raw[alias.get(s, s)] for s in symbols if alias.get(s, s) in raw}
        if not spots:
            return None, symbols
        asof = max(d for _, d in spots.values())  # the latest trading day in the data
        path = self.root / f"{asof:%Y-%m-%d}.parquet"
        done = set(pd.read_parquet(path, columns=["symbol"])["symbol"]) if path.exists() else set()
        todo = [s for s in symbols if s in spots and alias.get(s, s) not in done]
        progress(
            f"  as of {asof}: {len(spots)} priced, {len(done)} already recorded, {len(todo)} to do"
        )
        rows: list[dict] = []
        failed: list[str] = [s for s in symbols if s not in spots]
        empty = 0
        for n, s in enumerate(todo, 1):
            spot, _ = spots[s]
            try:
                r = summarize_chain(self.chain(s, spot, asof), spot, asof)
            except Exception as exc:
                progress(f"  {s}: failed ({exc})")
                failed.append(s)
                continue
            if not r:
                empty += 1
                r = [{"expiry": pd.NaT, "dte": 0, "spot": spot}]  # marks "no usable options"
            rows += [{"asof": pd.Timestamp(asof), "symbol": alias.get(s, s), **x} for x in r]
            if n % 100 == 0:
                progress(f"  {n}/{len(todo)} underlyings ({empty} without usable options)")
                self._save(path, rows)
                rows = []
        self._save(path, rows)
        return path, failed

    def _save(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        df["fetched_at"] = datetime.now(UTC)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.drop_duplicates(["symbol", "expiry"], keep="last").to_parquet(path)


def load_iv(root: str | Path = ROOT) -> pd.DataFrame:
    """All recorded days, sorted; rows without usable options are dropped."""
    files = sorted(Path(root).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df["expiry"].notna()].sort_values(["asof", "symbol", "expiry"]).reset_index(drop=True)
