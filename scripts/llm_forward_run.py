# ruff: noqa: E501
"""Forward test runner: find new S&P 1500 earnings releases, ask the LLM for probabilities BEFORE
the market reacts, and log them (research/preregistered/2026-11-07-llm-forward-test-v1.md).

    uv run python scripts/llm_forward_run.py                 # live: needs ANTHROPIC_API_KEY in .env
    uv run python scripts/llm_forward_run.py --dry-run       # plumbing only: no API calls, separate log

Run it twice on weekdays: about 08:45 ET (pre-open releases) and 16:45 ET (after-close releases).
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from atlas.factors.links import company_aliases
from atlas.llm_forward.baseline import PATH as BASELINE_PATH
from atlas.llm_forward.baseline import fit_baseline, load_baseline, score_baseline
from atlas.llm_forward.client import DEFAULT_MODEL, Completion, LLMClient
from atlas.llm_forward.prompt import PROMPT_SHA, PROMPT_V1, build_messages, parse_response
from atlas.llm_forward.session import is_eligible, reaction_session_start
from atlas.llm_forward.store import ROOT, PredictionStore
from atlas.lock import single_instance
from atlas.marketdata.earnings_releases import (
    EarningsReleaseDownloader,
    load_earnings_index,
    read_release,
)
from atlas.marketdata.filings import FilingClient
from atlas.marketdata.sec import ciks_from_files
from atlas.ml.llm_reader import MAX_CHARS, prepare_text
from atlas.net import install_default_timeout

START = pd.Timestamp("2026-11-08", tz="UTC")  # first filing date that may count
MODEL_CUTOFF = {
    "claude-sonnet-5-5": pd.Timestamp("2026-06-30", tz="UTC")
}  # documented, conservative
MONTHLY_CAP = float(os.getenv("ATLAS_LLM_MONTHLY_CAP", "75"))
CONSTITUENTS = [f"data/universes/{u}_constituents.csv" for u in ("sp500", "sp400", "sp600")]


class StubClient:
    """Dry run: no API call. Returns the base rates."""

    model = "dry-run"

    def complete(self, system: str, user: str) -> Completion:
        return Completion(
            '{"p_reaction_up": 0.5, "p_drift_up": 0.5, "p_big_move": 0.5, "rationale": "stub"}',
            0,
            0,
        )


def month_cost(store: PredictionStore, now: pd.Timestamp) -> float:
    df = store.records()
    if df.empty or "cost" not in df:
        return 0.0
    m = df[df["predicted_at"].str[:7] == f"{now:%Y-%m}"]
    return float(m["cost"].fillna(0).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--since-days", type=int, default=6, help="how far back to look for new releases"
    )
    ap.add_argument(
        "--ignore-start", action="store_true", help="dry run only: also process older releases"
    )
    ap.add_argument(
        "--pretend-fresh",
        action="store_true",
        help="dry run only: predicted_at = filing + 5 minutes",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    load_dotenv(".env")
    install_default_timeout()
    model = DEFAULT_MODEL
    store = PredictionStore(
        ROOT / ("predictions_dryrun.jsonl" if args.dry_run else "predictions.jsonl")
    )
    client = StubClient() if args.dry_run else LLMClient(model)
    if (args.ignore_start or args.pretend_fresh) and not args.dry_run:
        raise SystemExit("--ignore-start and --pretend-fresh are for dry runs only")
    now = pd.Timestamp.now(tz=UTC)
    cutoff = (
        max(START, MODEL_CUTOFF.get(model, START))
        if not args.dry_run
        else pd.Timestamp("2000-01-01", tz=UTC)
    )
    if args.ignore_start:
        cutoff = now - timedelta(days=args.since_days)
    with single_instance("llm_forward"):
        if not BASELINE_PATH.exists():
            print("fitting the frozen TF-IDF baseline (once) ...", flush=True)
            fit_baseline()
        bundle = load_baseline()
        symbols = set(Path("data/universes/sp1500.txt").read_text().split())
        ciks = {
            s: c
            for s, c in ciks_from_files([p for p in CONSTITUENTS if Path(p).exists()]).items()
            if s in symbols
        }
        if not args.no_refresh:
            since = (now - timedelta(days=args.since_days)).strftime("%Y-%m-%d")
            EarningsReleaseDownloader(FilingClient(os.environ["SEC_USER_AGENT"])).download(
                ciks, since=since
            )
        idx = load_earnings_index()
        idx["accepted_ts"] = pd.to_datetime(idx["accepted"], utc=True)
        cand = idx[(idx["accepted_ts"] >= cutoff) & idx["symbol"].isin(symbols)].sort_values(
            "accepted_ts"
        )
        cons = pd.concat(
            [pd.read_csv(p) for p in CONSTITUENTS if Path(p).exists()]
        ).drop_duplicates("Symbol")
        aliases = company_aliases(cons)
        done = spent = 0
        new = elig = unparsed = 0
        for r in cand.itertuples():
            if store.seen(r.accession, client.model):
                continue
            if args.limit and new >= args.limit:
                break
            new += 1
            predicted_at = (
                r.accepted_ts + pd.Timedelta(minutes=5)
                if args.pretend_fresh
                else pd.Timestamp.now(tz=UTC)
            )
            ok, why = is_eligible(r.accepted_ts, predicted_at)
            start = reaction_session_start(r.accepted_ts)
            rec = {
                "accession": r.accession, "cik": int(r.cik), "symbol": r.symbol,
                "accepted": r.accepted_ts.isoformat(), "filed": r.filed,
                "session_start": start.isoformat() if start is not None else None,
                "predicted_at": predicted_at.isoformat(), "eligible": ok, "reason": why,
                "model": client.model, "prompt_version": PROMPT_V1, "prompt_sha": PROMPT_SHA,
            }  # fmt: skip
            if ok:
                try:
                    raw = read_release("data/sec", r.cik, r.accession)
                except Exception as exc:
                    rec.update(eligible=False, reason=f"text unavailable: {str(exc)[:60]}")
                    store.append(rec)
                    continue
                names = aliases.get(r.symbol, [])
                text_llm = prepare_text(raw, names, r.symbol, 6000)
                rec["text_sha"] = __import__("hashlib").sha256(text_llm.encode()).hexdigest()
                rec.update(
                    score_baseline([prepare_text(raw, names, r.symbol, MAX_CHARS)], bundle)
                    .iloc[0]
                    .to_dict()
                )
                if month_cost(store, now) + spent >= MONTHLY_CAP:
                    rec.update(eligible=False, reason="monthly spend cap reached")
                else:
                    system, user = build_messages(text_llm)
                    out = client.complete(system, user)
                    spent += out.cost
                    parsed = parse_response(out.text)
                    rec.update(
                        raw_response=out.text,
                        input_tokens=out.input_tokens,
                        output_tokens=out.output_tokens,
                        cost=out.cost,
                    )
                    if parsed is None:
                        unparsed += 1
                        rec["reason"] = "unparseable reply"
                    else:
                        rec.update(parsed)
                        elig += 1
            store.append(rec)
            done += 1
        day = f"{now:%Y-%m-%d}"
        digest = store.write_digest(day) if not args.dry_run else None
        print(f"{'DRY RUN ' if args.dry_run else ''}new releases {new}: eligible and predicted {elig}, "
              f"unparseable {unparsed}, ineligible {new - elig - unparsed}; spent ${spent:.2f} "
              f"(month ${month_cost(store, now):.2f} of ${MONTHLY_CAP:.0f}); digest {digest}")  # fmt: skip


if __name__ == "__main__":
    main()
