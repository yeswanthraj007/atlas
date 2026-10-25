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
        [Adjustment.ALL, Adjustment.RAW, Adjustment.SPLIT]
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


TEMPLATE = '''"""Candidate signal: {name}.

Written BEFORE testing; never edited after its holdout opens (a change is a new trial).
"""

from __future__ import annotations

import pandas as pd

from atlas.lab.evaluate import CandidateSpec
from atlas.research import ResearchData

SPEC = CandidateSpec(
    name="{name}",
    hypothesis="TODO: what should happen, e.g. stocks with X outperform over the next month",
    rationale="TODO: why it should work: who is on the other side of the trade, and why "
    "they have not already removed the effect",
    horizon_days=21,
    author="{author}",
)


def compute(data: ResearchData) -> pd.DataFrame:
    """Score per stock per date (dates x "EQ:<ticker>"), higher = more attractive.
    Use only rows up to each date: the evaluator re-runs this on truncated data."""
    raise NotImplementedError
'''


def cmd_research_new(args: argparse.Namespace) -> int:
    path = Path("src/atlas/candidates") / f"{args.name}.py"
    if path.exists():
        print(f"{path} already exists", file=sys.stderr)
        return 2
    path.write_text(TEMPLATE.format(name=args.name, author=args.author))
    print(f"created {path}: write the hypothesis and rationale first, then compute()")
    return 0


def cmd_research_run(args: argparse.Namespace) -> int:
    from atlas.lab.evaluate import evaluate_candidate
    from atlas.research import load_research_data

    trial = evaluate_candidate(args.name, load_research_data(args.features))
    print(f"trial #{trial.trial_id}: {trial.status}")
    return 0 if trial.status == "promoted" else 1


def cmd_research_log(_: argparse.Namespace) -> int:
    from atlas.lab.trials import TrialLog

    log = TrialLog()
    trials = log.all()
    if not trials:
        print("no trials yet")
        return 0
    for t in trials:
        m = t.metrics
        print(
            f"#{t.trial_id:<3} {t.created_at[:10]} {t.candidate:28s} {t.status:18s} "
            f"dev t {m.get('dev_ic_t', float('nan')):+.2f}  holdout "
            f"{'opened' if t.holdout_opened else 'sealed'}  by {t.author}"
        )
    print(f"{len(trials)} trials, holdout opened {log.holdout_views()} times")
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
    d.add_argument("--adjustment", default="both", choices=["all", "raw", "split", "both"])
    d.set_defaults(fn=cmd_download)

    c = data.add_parser("check", help="run quality checks on stored bars")
    c.add_argument("symbols", nargs="*")
    c.add_argument("--file", action="append")
    c.set_defaults(fn=cmd_check)

    data.add_parser("factors", help="download Fama-French factors").set_defaults(fn=cmd_factors)
    data.add_parser("universe", help="list tradable US equities").set_defaults(fn=cmd_universe)

    research = sub.add_parser("research", help="signal research lab").add_subparsers(
        dest="cmd", required=True
    )
    rn = research.add_parser("new", help="scaffold a candidate signal")
    rn.add_argument("name")
    rn.add_argument("--author", default="claude")
    rn.set_defaults(fn=cmd_research_new)
    rr = research.add_parser("run", help="evaluate a candidate through the gates")
    rr.add_argument("name")
    rr.add_argument("--features", default="extended")
    rr.set_defaults(fn=cmd_research_run)
    research.add_parser("log", help="show every trial").set_defaults(fn=cmd_research_log)
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
