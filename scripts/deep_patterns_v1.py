"""Deep price-pattern models v1, exactly as pre-registered in
research/preregistered/2026-10-26-deep-patterns-v1.md.

    uv run python scripts/deep_patterns_v1.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge

from atlas.backtest.periodic import decile_aims, simulate
from atlas.lab.trials import log_external_trial
from atlas.ml.deep_patterns import (
    ChartCNN,
    PatchTransformer,
    TrainConfig,
    build_samples,
    predict,
    train,
)
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

PREREG = "research/preregistered/2026-10-26-deep-patterns-v1.md"
OUT = Path("data/reports/deep_patterns_v1")
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
VAL_YEAR = 2023
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "COST", "JPM", "XOM", "INTC", "SNDK"]


def evaluate(
    name: str,
    prob: np.ndarray,
    keys: pd.DataFrame,
    fwd: np.ndarray,
    cost: pd.DataFrame,
    sp500: set[str],
) -> dict:
    df = keys.assign(p=prob, r=fwd).dropna(subset=["r"])
    dates = sorted(df["date"].unique())[::4]  # monthly, non-overlapping 20-day outcomes
    m = df[df["date"].isin(dates)]
    ic = m.groupby("date").apply(lambda g: stats.spearmanr(g["p"], g["r"]).statistic)
    t = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))
    med = m.groupby("date")["r"].transform("median")
    acc = float(((m["p"] > 0.5) == (m["r"] > med)).mean())
    seg = {}
    for label, mask in (
        ("sp500", m["instrument"].isin(sp500)),
        ("mid_small", ~m["instrument"].isin(sp500)),
    ):
        s = (
            m[mask]
            .groupby("date")
            .apply(lambda g: stats.spearmanr(g["p"], g["r"]).statistic if len(g) > 30 else np.nan)
            .dropna()
        )
        seg[label] = {
            "ic": float(s.mean()),
            "t": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))),
        }
    pred = m.pivot(index="date", columns="instrument", values="p")
    fwd_w = m.pivot(index="date", columns="instrument", values="r").fillna(0.0)
    c = cost.reindex(index=pred.index, columns=pred.columns)
    ls = {tau: simulate(decile_aims(pred), fwd_w, c, tau) for tau in (1.0, 0.5)}
    res = {
        "model": name,
        "ic": float(ic.mean()),
        "ic_t": t,
        "months": len(ic),
        "accuracy_vs_median": acc,
        "segments": seg,
        "ls_gross": float(ls[1.0].gross.mean()),
        "ls_net_tau1": float(ls[1.0].net.mean()),
        "ls_net_tau05": float(ls[0.5].net.mean()),
        "turnover_tau05": float(ls[0.5].turnover.mean()),
    }
    per = {}
    for tk in TICKERS:
        d = m[m["instrument"] == f"EQ:{tk}"]
        if len(d):
            dmed = med[d.index]
            per[tk] = {"months": len(d), "right": float(((d["p"] > 0.5) == (d["r"] > dmed)).mean())}
    res["per_ticker"] = per
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pre-registration: {PREREG}")
    rd = load_research_data("price", universe="sp1500")
    dates = rd.closes.index
    sp500 = {f"EQ:{s}" for s in Path("data/universes/sp500.txt").read_text().split()}
    t0 = time.time()
    s = build_samples(rd.bars, dates, stride=5)
    print(f"{len(s.keys):,} samples from {rd.closes.shape[1]} stocks in {time.time() - t0:.0f}s")

    d = pd.DatetimeIndex(s.keys["date"])
    known = np.isfinite(s.target)
    tr = np.where(known & (s.label_end <= TRAIN_END))[0]
    va = np.where(known & (d.year == VAL_YEAR) & (s.label_end.year == VAL_YEAR))[0]
    te = np.where((d >= TEST_START) & np.isfinite(s.fwd_return))[0]
    print(f"train {len(tr):,} | validation (2023) {len(va):,} | test (2024+) {len(te):,}")

    results = []
    for name, model, kind, lr in [
        ("chart_cnn", ChartCNN(), "cnn", 1e-3),
        ("patch_transformer", PatchTransformer(), "seq", 3e-4),
    ]:
        print(f"\ntraining {name} ({sum(p.numel() for p in model.parameters()):,} parameters)")
        t0 = time.time()
        cfg = TrainConfig(lr=lr)
        model = train(model, kind, s, tr, va, cfg)
        torch.save(model.state_dict(), OUT / f"{name}.pt")
        prob = predict(model, kind, s, te)
        r = evaluate(
            name, prob, s.keys.iloc[te].reset_index(drop=True), s.fwd_return[te], rd.spread, sp500
        )
        r["epochs"] = len(cfg.history)
        r["train_minutes"] = (time.time() - t0) / 60
        results.append(r)

    # baseline: ridge on simple price features, same samples and splits
    c = rd.closes
    feats = {
        "reversal": -(c / c.shift(21) - 1),
        "momentum": c.shift(21) / c.shift(252) - 1,
        "volatility": c.pct_change().rolling(252).std(),
    }
    x = np.column_stack(
        [
            f.rank(axis=1, pct=True)
            .stack(future_stack=True)
            .reindex(pd.MultiIndex.from_frame(s.keys))
            .fillna(0.5)
            .to_numpy()
            - 0.5
            for f in feats.values()
        ]
    )
    ridge = Ridge(alpha=10.0).fit(x[tr], s.target[tr])
    results.append(
        evaluate(
            "ridge_baseline",
            ridge.predict(x[te]),
            s.keys.iloc[te].reset_index(drop=True),
            s.fwd_return[te],
            rd.spread,
            sp500,
        )
    )

    base_ic = results[-1]["ic"]
    print("\n=== Frozen test, 2024-2026, S&P 1500 (monthly, non-overlapping) ===")
    for r in results:
        print(
            f"\n{r['model']}: IC {r['ic']:+.4f} (t {r['ic_t']:+.2f}, {r['months']} months) | "
            f"beats-median accuracy {r['accuracy_vs_median']:.1%}"
        )
        print(
            f"  S&P 500 IC {r['segments']['sp500']['ic']:+.4f} (t "
            f"{r['segments']['sp500']['t']:+.2f}) | mid/small IC "
            f"{r['segments']['mid_small']['ic']:+.4f} (t {r['segments']['mid_small']['t']:+.2f})"
        )
        print(
            f"  top-minus-bottom decile per month: gross {r['ls_gross']:+.2%}, net (tau 1) "
            f"{r['ls_net_tau1']:+.2%}, net (tau 0.5) {r['ls_net_tau05']:+.2%}"
        )
        print(
            "  per ticker (right vs median): "
            + ", ".join(f"{k} {v['right']:.0%}" for k, v in r["per_ticker"].items())
        )
        if r["model"] != "ridge_baseline":
            checks = {
                "IC t >= 2": r["ic_t"] >= 2.0 and r["ic"] > 0,
                "net decile spread > 0 (tau 0.5)": r["ls_net_tau05"] > 0,
                "beats ridge baseline IC": r["ic"] > base_ic,
            }
            passed = all(checks.values())
            for k, ok in checks.items():
                print(f"  {'PASS' if ok else 'FAIL'}  {k}")
            trial = log_external_trial(
                f"deep_{r['model']}",
                "promoted" if passed else "rejected_holdout",
                f"{r['model']} on raw price history predicts next-20-day relative returns",
                "Jiang, Kelly & Xiu (2023) chart CNN; PatchTST-style transformer",
                r,
                True,
                __file__,
                notes=f"pre-registered in {PREREG}",
            )
            print(f"  -> trial #{trial.trial_id}: {'PROMOTED' if passed else 'rejected'}")
            if passed:
                reg = Registry.load()
                reg.record(
                    SignalRecord(
                        f"deep_{r['model']}",
                        "deep price-pattern model",
                        "direction",
                        "ic_20d",
                        r["ic"],
                        r["ic_t"],
                        "frozen 2024-2026",
                        f"trial #{trial.trial_id}",
                    )
                )
                reg.save()
    log_external_trial(
        "deep_ridge_baseline",
        "rejected_holdout",
        "ridge baseline",
        "comparison for deep models",
        results[-1],
        True,
        __file__,
        notes="baseline; not eligible for promotion",
    )
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in results]).to_csv(
        OUT / "results.csv", index=False
    )


if __name__ == "__main__":
    main()
