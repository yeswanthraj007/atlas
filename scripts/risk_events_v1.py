"""Risk and event predictors v1, exactly as pre-registered in
research/preregistered/2026-11-03-risk-events-v1.md.

    uv run python scripts/risk_events_v1.py            # full run (opens the 2024-2026 test)
    uv run python scripts/risk_events_v1.py --dev      # validation 2023 only, no test metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from atlas.lab.trials import log_external_trial
from atlas.ml.risk_events import (
    BASE_FEATURES,
    EVENT_FEATURES,
    WIKI_FEATURES,
    add_event_ratios,
    build_drawdown_samples,
    build_events,
    choose_and_fit,
    daily_features,
    days_since_event_panel,
    paired_t,
    predict_above_median,
    score_task,
    take,
    wiki_features,
)
from atlas.ml.volatility import forward_log_vol
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

PREREG = "research/preregistered/2026-11-03-risk-events-v1.md"
OUT = Path("data/reports/risk_events_v1")
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-09-30", tz="UTC")


def log(msg: str) -> None:
    print(msg, flush=True)


def split_masks(date: pd.Series, label_date: pd.Series, y: np.ndarray):
    """Train: outcome known by 2022-12-31. Validation: starts in 2023, known by 2023-12-31.
    Test: starts 2024-01-01 .. 2026-09-30."""
    ok = np.isfinite(y)
    tr = ok & (label_date <= TRAIN_END).to_numpy()
    va = ok & (date.dt.year == 2023).to_numpy() & (label_date <= VAL_END).to_numpy()
    te = ok & (date >= TEST_START).to_numpy() & (date <= TEST_END).to_numpy()
    return tr, va, te


def run_task(name, x_cols, rows, y, baselines_fn, group, dev, log_trials, wiki_cols):
    tr, va, te = split_masks(rows["date"], rows["label_date"], y)
    log(f"\n== {name}: train {tr.sum():,} | validation {va.sum():,} | test {te.sum():,}")
    out = {}
    for variant, cols in (("with_wiki", x_cols + wiki_cols), ("no_wiki", x_cols)):
        x = rows[cols]
        model, best, scores = choose_and_fit(x[tr], y[tr], x[va], y[va], x[tr | va], y[tr | va])
        log(
            f"   {variant}: chosen (depth, iterations) {best}; validation log loss "
            f"{ {str(k): round(v, 4) for k, v in scores.items()} }"
        )
        if dev:
            continue
        prob = np.full(len(rows), np.nan)
        prob[te] = model.predict_proba(x[te])[:, 1]
        month = rows["date"].dt.strftime("%Y-%m").to_numpy()
        res = score_task(
            f"{name}_{variant}", prob[te], y[te], month[te],
            {k: v[te] for k, v in baselines_fn(rows, te, tr, group).items()},
        )  # fmt: skip
        out[variant] = (res, best)
    return out


def report(name, out, log_trials):
    full, best = out["with_wiki"]
    plain, _ = out["no_wiki"]
    wiki_t = paired_t(full.monthly, plain.monthly)
    log(
        f"\n{name}: accuracy {full.accuracy:.1%} (AUC {full.auc:.3f}, {full.n:,} test rows, "
        f"{full.months} months) | most confident 20% each side: {full.confident_accuracy:.1%}"
    )
    for k, v in full.baseline_accuracy.items():
        log(f"   baseline {k}: {v:.1%}")
    log(
        f"   margin over best baseline ({full.best_baseline}): {full.margin:+.1%} "
        f"(paired monthly t {full.margin_t:+.2f})"
    )
    log(
        f"   without Wikipedia: {plain.accuracy:.1%} | attention adds "
        f"{full.accuracy - plain.accuracy:+.2%} (paired t {wiki_t:+.2f})"
    )
    a = full.accuracy >= 0.65
    b = full.margin > 0 and full.margin_t >= 3.0
    promoted = b and full.accuracy >= 0.60
    log(f"   {'PASS' if a else 'FAIL'}  A: accuracy >= 65%")
    log(f"   {'PASS' if b else 'FAIL'}  B: beats the best naive baseline, paired t >= 3.0")
    verdict = (
        "PROMOTED"
        if promoted
        else "reaches 65% but adds nothing over the naive baseline"
        if a
        else "rejected"
    )
    log(f"   VERDICT: {verdict}")
    if log_trials:
        metrics = dict(accuracy=full.accuracy, auc=full.auc, n=full.n, months=full.months,
                       confident_accuracy=full.confident_accuracy, margin=full.margin,
                       margin_t=full.margin_t, best_baseline=full.best_baseline,
                       baseline_accuracy=full.baseline_accuracy, no_wiki_accuracy=plain.accuracy,
                       wiki_margin_t=wiki_t, chosen=list(best))  # fmt: skip
        t = log_external_trial(
            f"risk_{name}", "promoted" if promoted else "rejected_holdout",
            f"{name} is predictable above the naive baseline", "risk/event predictors v1",
            metrics, True, __file__, notes=f"pre-registered in {PREREG}")  # fmt: skip
        log(f"   -> trial #{t.trial_id}")
        if promoted:
            reg = Registry.load()
            reg.record(SignalRecord(f"risk_{name}", f"{name} classifier", "risk",
                                    "accuracy_vs_baseline_t", full.accuracy, full.margin_t,
                                    "frozen 2024-2026", f"trial #{t.trial_id}"))  # fmt: skip
            reg.save()


def main(dev: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log(
        f"pre-registration: {PREREG}"
        + ("  [DEV: validation only, test metrics not computed]" if dev else "")
    )
    rd = load_research_data("price", universe="sp1500")
    closes, idx = rd.closes, rd.closes.index
    volume = pd.DataFrame({k: v["volume"] for k, v in rd.bars.items()}).reindex(idx)[closes.columns]
    earnings = pd.read_parquet("data/sec/earnings_index.parquet")
    ev = build_events(closes, rd.benchmark, earnings)
    dse = days_since_event_panel(idx, closes.columns, ev[["instrument", "d0"]])
    panels = daily_features(closes, volume, rd.split_closes, rd.benchmark, dse)
    wiki_path = Path("data/social/wikipedia/views.parquet")
    wiki_cols: list[str] = []
    if wiki_path.exists():
        panels.update(wiki_features(pd.read_parquet(wiki_path), idx, closes.columns))
        wiki_cols = WIKI_FEATURES
    else:
        log("WARNING: no Wikipedia views; running without the attention group")
    n_wiki = int(np.isfinite(panels["wiki_att"][-1]).sum()) if wiki_cols else 0
    log(
        f"{len(ev):,} earnings events, {len(closes.columns)} stocks, "
        f"Wikipedia attention for {n_wiki} stocks"
    )

    # ---------------- E1 / E2 ----------------
    p, j = ev["p"].to_numpy(), ev["stock"].to_numpy()
    mem = rd.membership.reindex(index=idx, columns=closes.columns).fillna(False).to_numpy()
    ev = ev[mem[p, j]].reset_index(drop=True)
    p, j = ev["p"].to_numpy(), ev["stock"].to_numpy()
    ev = add_event_ratios(ev, panels["lrv_63"])
    xe = take(panels, BASE_FEATURES + wiki_cols, p, j)
    ev = pd.concat([ev, xe], axis=1)
    ev["label_date"] = ev["date"]  # the outcome is known on the reaction day
    ev["quarter"] = (ev["accepted"].dt.year * 10 + ev["accepted"].dt.quarter).to_numpy()
    vol63 = ev["lrv_63"].to_numpy()

    def e1_baselines(rows, te, tr, group):
        q = rows["quarter"].to_numpy()
        own = rows["med8"].fillna(rows["med8"].median()).to_numpy()
        return {
            "own_median_move": _pad(te, predict_above_median(own[te], q[te])),
            "vol63_rank": _pad(te, predict_above_median(vol63[te], q[te])),
        }

    def e2_baselines(rows, te, tr, group):
        q = rows["quarter"].to_numpy()
        last_above = (rows["prev1"] > rows["med8"]).to_numpy().astype(float)
        ratio = (rows["lrv_5"] - rows["lrv_63"]).to_numpy()
        return {
            "last_event_above_median": _pad(te, last_above[te]),
            "vol5_over_vol63_rank": _pad(te, predict_above_median(ratio[te], q[te])),
        }

    ev_cols = BASE_FEATURES + EVENT_FEATURES
    for name, ycol, bl in (("E1", "y_e1", e1_baselines), ("E2", "y_e2", e2_baselines)):
        out = run_task(name, ev_cols, ev, ev[ycol].to_numpy(), bl, None, dev, not dev, wiki_cols)
        if not dev:
            report(name, out, True)

    # ---------------- D1 ----------------
    s = build_drawdown_samples(closes, rd.membership)
    xs = take(panels, BASE_FEATURES + wiki_cols, s["p"].to_numpy(), s["stock"].to_numpy())
    s = pd.concat([s, xs], axis=1)
    # HAR baseline: OLS of the realised 21-day volatility on lrv_5, lrv_21, lrv_63, fit on training
    fv = forward_log_vol(closes, 21).to_numpy()
    s["fwd_lv"] = fv[s["p"].to_numpy(), s["stock"].to_numpy()]
    tr_d, _, _ = split_masks(s["date"], s["label_date"], s["y_d1"].to_numpy())
    har_x = ["lrv_5", "lrv_21", "lrv_63"]
    fit = tr_d & s[har_x].notna().all(axis=1).to_numpy() & s["fwd_lv"].notna().to_numpy()
    har = LinearRegression().fit(s.loc[fit, har_x], s.loc[fit, "fwd_lv"])
    har_pred = har.predict(s[har_x].fillna(s[har_x].median()))
    g = s["p"].to_numpy()

    def d1_baselines(rows, te, tr, group):
        return {
            "vol63_rank": _pad(
                te, predict_above_median(rows["lrv_63"].fillna(-2).to_numpy()[te], g[te])
            ),
            "vol21_rank": _pad(
                te, predict_above_median(rows["lrv_21"].fillna(-2).to_numpy()[te], g[te])
            ),
            "har_forecast_rank": _pad(te, predict_above_median(har_pred[te], g[te])),
        }

    out = run_task("D1", BASE_FEATURES, s, s["y_d1"].to_numpy(), d1_baselines, None, dev, not dev,
                   wiki_cols)  # fmt: skip
    if not dev:
        report("D1", out, True)
        log("\ndone")
    (OUT / "ran.json").write_text(json.dumps({"dev": dev}))


def _pad(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place values for the masked rows back into a full-length array (NaN elsewhere)."""
    out = np.full(len(mask), np.nan)
    out[mask] = values
    return out


if __name__ == "__main__":
    main(dev="--dev" in sys.argv)
