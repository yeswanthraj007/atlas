"""Command-line entry point:  uv run atlas <command> ...

atlas data download AAPL MSFT SPY --start 2016-01-01
atlas data check
atlas data factors
atlas data universe
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from atlas.config import MissingCredentialsError, load_settings
from atlas.marketdata.bars import Adjustment, Timeframe
from atlas.marketdata.quality import check_bars, has_errors, report
from atlas.marketdata.store import BarStore


def _date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _symbols(args: argparse.Namespace) -> list[str]:
    syms = [s.upper() for s in args.symbols]
    for f in getattr(args, "file", None) or []:
        syms += [
            ln.strip().upper()
            for ln in Path(f).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    return sorted(set(syms))


def cmd_download(args: argparse.Namespace) -> int:
    from atlas.marketdata.alpaca import AlpacaBarProvider

    settings = load_settings()
    store = BarStore(settings.data_dir)
    provider = AlpacaBarProvider(settings)
    symbols = _symbols(args)
    if not symbols:
        print("no symbols given", file=sys.stderr)
        return 2
    end = _date(args.end) if args.end else datetime.now(UTC)
    tf = Timeframe(args.timeframe)
    adjustments = (
        [Adjustment.ALL, Adjustment.RAW]
        if args.adjustment == "both"
        else [Adjustment(args.adjustment)]
    )
    issues = []
    for adj in adjustments:
        print(
            f"downloading {len(symbols)} symbols, {tf}, adjustment={adj}, "
            f"feed={settings.alpaca_historical_feed}"
        )
        bars = provider.get_bars(symbols, _date(args.start), end, tf, adj)
        for sym, df in bars.items():
            store.write(sym, df, tf, adj)
            if adj is Adjustment.ALL:
                issues += check_bars(sym, df, daily=tf is Timeframe.DAY)
        missing = sorted(set(symbols) - set(bars))
        print(f"  stored {len(bars)}; no data for {len(missing)}: {missing[:20]}")
    if issues:
        print(report(issues).to_string(index=False))
    return 1 if has_errors(issues) else 0


def cmd_check(args: argparse.Namespace) -> int:
    store = BarStore(load_settings().data_dir)
    symbols = _symbols(args) or store.symbols()
    issues = []
    for sym in symbols:
        issues += check_bars(sym, store.read(sym))
    print(report(issues).to_string(index=False) if issues else f"{len(symbols)} symbols clean")
    return 1 if has_errors(issues) else 0


def cmd_factors(_: argparse.Namespace) -> int:
    from atlas.marketdata.fama_french import fetch_daily_factors

    df = fetch_daily_factors(load_settings().data_dir / "factors")
    print(f"{len(df)} days, {df.index[0].date()} to {df.index[-1].date()}: {list(df.columns)}")
    return 0


def cmd_universe(_: argparse.Namespace) -> int:
    from atlas.marketdata.alpaca import AlpacaUniverse

    settings = load_settings()
    assets = AlpacaUniverse(settings).assets()
    path = settings.data_dir / "universe.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([a.__dict__ for a in assets]).to_csv(path, index=False)
    print(f"{len(assets)} tradable US equities written to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlas")
    sub = p.add_subparsers(dest="group", required=True)
    data = sub.add_parser("data", help="market data").add_subparsers(dest="cmd", required=True)

    d = data.add_parser("download", help="download bars from Alpaca")
    d.add_argument("symbols", nargs="*")
    d.add_argument("--file", action="append", help="file with one symbol per line (repeatable)")
    d.add_argument("--start", default="2016-01-01")
    d.add_argument("--end")
    d.add_argument("--timeframe", default="1d", choices=[t.value for t in Timeframe])
    d.add_argument("--adjustment", default="both", choices=["all", "raw", "both"])
    d.set_defaults(fn=cmd_download)

    c = data.add_parser("check", help="run quality checks on stored bars")
    c.add_argument("symbols", nargs="*")
    c.add_argument("--file", action="append")
    c.set_defaults(fn=cmd_check)

    data.add_parser("factors", help="download Fama-French factors").set_defaults(fn=cmd_factors)
    data.add_parser("universe", help="list tradable US equities").set_defaults(fn=cmd_universe)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except MissingCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
