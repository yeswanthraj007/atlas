"""Ensemble stage of the forecaster shootout (Amendment 2 of
research/preregistered/2026-11-07-forecaster-shootout-v1.md). Reads the saved predictions.

    uv run python scripts/forecaster_ensemble_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from atlas.lab.trials import log_external_trial
from atlas.ml.ensemble import returns_stage, vol_stage
from atlas.recommend.engine import Registry, SignalRecord

PREREG = "research/preregistered/2026-11-07-forecaster-shootout-v1.md"
MODELS = ["arima_garch", "lstm", "marketgpt", "chronos", "timesfm"]
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")


def main() -> None:
    path = sys.argv[sys.argv.index("--predictions") + 1] if "--predictions" in sys.argv else None
    note = " (after the ARIMA-GARCH fix)" if path else ""
    df = pd.read_parquet(Path(path or "data/reports/shootout_v1/predictions.parquet"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    r, v = returns_stage(df, MODELS, SPLIT), vol_stage(df, MODELS, SPLIT)
    print(f"selection window: dates before {SPLIT.date()}; evaluation: {r['eval_dates']} dates, "
          f"{r['eval_rows']:,} stock-dates")  # fmt: skip

    print("\n=== Returns (evaluation window; models chosen on the selection window) ===")
    print(f"selected top 3: {r['selected']} | best single: {r['best_single']}")
    for n, ic in r["eval_ic"].items():
        extra = f" | t vs ridge {r['t_vs_ridge'][n]:+.2f}" if n != "ridge" else ""
        print(f"  {n:12s} IC {ic:+.4f} (t {r['eval_t'][n]:+.2f}){extra}")
    for n, t in r["t_vs_best_single"].items():
        print(f"  {n} vs best single ({r['best_single']}): paired t {t:+.2f}")

    print("\n=== Volatility (calibrated on the selection window) ===")
    print(f"selected top 3: {v['selected']} | best single: {v['best_single']}")
    for n, q in v["eval_qlike"].items():
        extra = f" | DM t vs HAR {v['dm_t_vs_har'][n]:+.2f}" if n != "har" else ""
        print(f"  {n:12s} QLIKE {q:.4f} | accuracy {v['eval_accuracy'][n]:.1%}{extra}")
    for n, t in v["t_vs_best_single"].items():
        print(f"  {n} vs best single ({v['best_single']}): paired t {t:+.2f}")

    reg = Registry.load()
    for name in ("ENS-TOP3", "ENS-ALL"):
        ur = r["eval_ic"][name] > r["eval_ic"]["ridge"] and r["t_vs_ridge"][name] >= 3.0
        uv = v["eval_qlike"][name] < v["eval_qlike"]["har"] and v["dm_t_vs_har"][name] >= 3.0
        better_r = r["t_vs_best_single"][name] >= 2.0
        better_v = v["t_vs_best_single"][name] >= 2.0
        yes = lambda ok: "YES" if ok else "no"  # noqa: E731
        print(
            f"\n{name}: useful for returns {yes(ur)}, for volatility {yes(uv)}; "
            f"beats best single: returns {yes(better_r)}, volatility {yes(better_v)}"
        )
        metrics = {
            "returns": {k: x[name] for k, x in r.items() if isinstance(x, dict) and name in x},
            "volatility": {k: x[name] for k, x in v.items() if isinstance(x, dict) and name in x},
            "selected_returns": r["selected"],
            "selected_vol": v["selected"],
            "eval_dates": r["eval_dates"],
        }
        t = log_external_trial(
            f"shootout_{name.lower()}", "promoted" if (ur or uv) else "rejected_holdout",
            f"{name} of the five forecasters beats the baselines and the best single model",
            "forecast combination; members chosen on 2024 only", metrics, True, __file__,
            notes=f"pre-registered in {PREREG} (Amendment 2){note}")  # fmt: skip
        print(f"  -> trial #{t.trial_id}")
        if uv:
            reg.record(SignalRecord(f"shootout_{name.lower()}_vol", f"{name} 20-day volatility",
                                    "risk", "qlike_dm_t_vs_har", v["eval_qlike"][name],
                                    v["dm_t_vs_har"][name], "evaluation 2025-2026",
                                    f"trial #{t.trial_id}"))  # fmt: skip
        if ur:
            reg.record(SignalRecord(f"shootout_{name.lower()}_ret", f"{name} 20-day returns",
                                    "direction", "ic_20d", r["eval_ic"][name], r["eval_t"][name],
                                    "evaluation 2025-2026", f"trial #{t.trial_id}"))  # fmt: skip
    reg.save()


if __name__ == "__main__":
    main()
