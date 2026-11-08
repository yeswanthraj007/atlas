# ruff: noqa: E501
"""Score the forward test (pre-registered checkpoints at 600, 1,500 and 3,000 eligible predictions
with a complete 20-day outcome). Reads the append-only log; never changes it.

    uv run python scripts/llm_forward_score.py              # live log
    uv run python scripts/llm_forward_score.py --dry-run --force   # plumbing check

Price data must reach 20 trading days past the releases: refresh the S&P 1500 bars first.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from atlas.lab.trials import log_external_trial
from atlas.llm_forward.baseline import load_baseline
from atlas.llm_forward.client import DEFAULT_MODEL
from atlas.llm_forward.score import compute_outcomes, score_stream
from atlas.llm_forward.store import ROOT, PredictionStore
from atlas.ml.risk_events import build_events
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

PREREG = "research/preregistered/2026-11-07-llm-forward-test-v1.md"
CHECKPOINTS = (600, 1500, 3000)
STATE = ROOT / "checkpoints.json"


def main() -> None:
    dry, force = "--dry-run" in sys.argv, "--force" in sys.argv
    store = PredictionStore(ROOT / ("predictions_dryrun.jsonl" if dry else "predictions.jsonl"))
    rec = store.records()
    if rec.empty:
        print("no predictions logged yet")
        return
    model = "dry-run" if dry else DEFAULT_MODEL
    rec = rec[
        (rec["model"] == model)
        & rec["eligible"].fillna(False).astype(bool)
        & rec["p_big_move"].notna()
    ].copy()
    print(f"{len(rec):,} eligible, parsed predictions from {model}")
    if rec.empty:
        return
    rd = load_research_data("price", universe="sp1500")
    closes, bench = rd.closes, rd.benchmark
    print(f"price data ends {closes.index[-1].date()}")
    rec["accepted"] = pd.to_datetime(rec["accepted"], utc=True)
    rec["instrument"] = "EQ:" + rec["symbol"]
    rec = rec[rec["instrument"].isin(closes.columns)].reset_index(drop=True)
    out = compute_outcomes(rec["accepted"], rec["instrument"], closes, bench)
    rec = pd.concat([rec, out], axis=1)
    ev = build_events(closes, bench, pd.read_parquet("data/sec/earnings_index.parquet"))
    qmed = (
        ev.groupby(ev["accepted"].dt.year * 10 + ev["accepted"].dt.quarter)["abs_react"]
        .median()
        .dropna()
        .to_dict()
    )
    rec["quarter"] = (rec["accepted"].dt.year * 10 + rec["accepted"].dt.quarter).astype(int)
    rec["month"] = rec["accepted"].dt.strftime("%Y-%m")
    rec["day"] = rec["accepted"].dt.strftime("%Y-%m-%d")
    p = (rec["d0"] - 1).clip(lower=0).to_numpy()
    rec["cost"] = (
        pd.Series(rd.spread.to_numpy()[p, closes.columns.get_indexer(rec["instrument"])])
        .fillna(0.002)
        .to_numpy()
    )
    complete = int(rec["drift"].notna().sum())
    print(
        f"{int(rec['react'].notna().sum())} with a reaction outcome, {complete} with a complete 20-day outcome"
    )
    nxt = [c for c in CHECKPOINTS if c > complete]
    done = set(json.loads(STATE.read_text())) if STATE.exists() and not dry else set()
    due = [c for c in CHECKPOINTS if c <= complete and c not in done]
    if not due and not force:
        print(
            f"next checkpoint at {nxt[0] if nxt else 'none'} complete outcomes (now {complete}); not scoring yet"
        )
        return
    s = score_stream(rec, qmed, load_baseline()["base_rates"])
    if not s:
        print("nothing scoreable yet")
        return
    print(json.dumps(s, indent=2, default=float))
    if dry or not due:
        return
    t3, t2 = s["t3"], s.get("t2", {})
    promoted = (
        t2.get("ic_monthly_t", 0) >= 3
        and t2.get("spread_net", -1) > 0
        and t2.get("ic_all", 0) > t2.get("ic_tfidf_all", 0)
    ) or (
        t3["llm"]["log_loss"] < t3["tfidf"]["log_loss"]
        and t3["paired_t_logloss_llm_minus_tfidf"] <= -3
    )
    cp = max(due)
    t = log_external_trial(f"llm_forward_{model}_n{cp}", "promoted" if promoted else "rejected_holdout",
                           "a frontier LLM reading masked earnings releases beats the baselines going forward",
                           "forward test, predictions logged before outcomes", s, True, __file__, notes=f"pre-registered in {PREREG}")  # fmt: skip
    print(f"checkpoint {cp}: trial #{t.trial_id}: {'PROMOTED' if promoted else 'rejected'}")
    STATE.write_text(json.dumps(sorted(done | set(due))))
    if promoted:
        reg = Registry.load()
        reg.record(
            SignalRecord(
                "llm_forward_drift",
                f"{model} reading earnings releases",
                "direction",
                "ic_20d",
                t2.get("ic_monthly_mean", float("nan")),
                t2.get("ic_monthly_t", float("nan")),
                f"forward test, n={cp}",
                f"trial #{t.trial_id}",
            )
        )
        reg.save()


if __name__ == "__main__":
    main()
