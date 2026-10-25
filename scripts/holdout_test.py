"""Frozen-model test: train on 2016-2024 only, then predict every month of 2025-2026.

    uv run python scripts/holdout_test.py --tickers AAPL NVDA TSLA COST JPM XOM

Steps
  1. Training rows: sample dates whose 21-day outcome was known by --train-end.
  2. Model choice: purged time-series CV on training rows only.
  3. Calibration: a second model trained on data before --calib-start predicts
     --calib-start..--train-end; the realised returns per prediction decile turn a
     predicted rank into an expected return. Uses training-period data only.
  4. The chosen model is fit once on all training rows and FROZEN.
  5. Every 21 trading days after --train-end it predicts the next 21 days; each
     prediction is scored when its outcome is known. No retraining, no tuning.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atlas.ml.dataset import build_dataset, feature_columns
from atlas.ml.models import GBMRanker, RidgeRanker, cross_validate, daily_ic
from atlas.research import load_research_data

MODELS = {
    "ridge": lambda: RidgeRanker(alpha=10.0),
    "gbm_shallow": lambda: GBMRanker(max_leaf_nodes=7, min_samples_leaf=800),
    "gbm": lambda: GBMRanker(max_leaf_nodes=15, min_samples_leaf=400),
}
HORIZON = 21
OUT = Path("data/reports/holdout")


def t_vs(values: pd.Series, null: float) -> tuple[float, float]:
    """Mean and t-stat of monthly values against a null (months are the independent unit:
    stocks within a month are correlated, so a per-stock binomial test would overstate)."""
    v = values.dropna()
    return float(v.mean()), float((v.mean() - null) / (v.std(ddof=1) / np.sqrt(len(v))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--calib-start", default="2023-01-01")
    ap.add_argument("--features", default="all", choices=["price", "all", "extended"])
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "NVDA", "TSLA", "COST", "JPM", "XOM"])
    args = ap.parse_args()
    args.tickers = [t.upper() for x in args.tickers for t in re.split(r"[\s,]+", x) if t]
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    calib_start = pd.Timestamp(args.calib_start, tz="UTC")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"feature set: {args.features}")

    rd = load_research_data(args.features)
    closes, split = rd.closes, rd.split_closes
    idx = closes.index
    train_grid = idx[(idx >= idx[0] + pd.Timedelta(days=380)) & (idx <= train_end)][::HORIZON]
    after = idx[idx > train_end]
    test_dates = after[::HORIZON]
    test_dates = test_dates[idx.get_indexer(test_dates) + HORIZON < len(idx)]  # outcome known
    data = build_dataset(
        rd.factors, closes, train_grid.union(test_dates), HORIZON, rd.membership, rd.benchmark
    )
    feats = feature_columns(data)
    fwd_raw = (closes.shift(-HORIZON) / closes - 1.0).stack(future_stack=True)
    data["fwd_return"] = fwd_raw.reindex(data.index).to_numpy()
    dates = data.index.get_level_values("date")
    label_end = pd.to_datetime(data["label_end"])

    train = data[(label_end <= train_end) & data["target"].notna()]
    test = data[dates.isin(test_dates)]
    print(
        f"train: {train.index.get_level_values('date').min().date()} .. "
        f"{train_end.date()} | {len(train):,} rows"
    )
    print(
        f"test : {test_dates[0].date()} .. {test_dates[-1].date()} (+21 days) | "
        f"{len(test_dates)} months | {len(test):,} predictions | model FROZEN\n"
    )

    cv = {k: cross_validate(m, train, feats) for k, m in MODELS.items()}
    best = max(cv, key=lambda k: cv[k])
    print(
        "model choice by purged CV on training data only:",
        ", ".join(f"{k} IC {v:.4f}" for k, v in cv.items()),
        f"-> {best}",
    )

    # Calibration on the last training years, using a model that never saw them
    cal_train = train[pd.to_datetime(train["label_end"]) < calib_start]
    cal_rows = train[train.index.get_level_values("date") >= calib_start]
    cal_model = MODELS[best]().fit(cal_train[feats], cal_train["target"])
    cal = cal_rows.assign(pred=cal_model.predict(cal_rows[feats]))
    cal["decile"] = cal.groupby(level="date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)
    )
    decile_ret = cal.groupby("decile")["fwd_return"].mean()

    model = MODELS[best]().fit(train[feats], train["target"])  # frozen from here on
    test = test.assign(pred=model.predict(test[feats]))
    scored = test[test["target"].notna()].copy()
    scored["decile"] = scored.groupby(level="date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)
    )
    scored["expected_return"] = scored["decile"].map(decile_ret)

    # ---------------- 1. accuracy across all stocks ----------------
    g = scored.groupby(level="date")
    beat_acc = g.apply(
        lambda d: float(((d["pred"] > d["pred"].median()) == (d["target"] > 0)).mean())
    )
    top_prec = g.apply(
        lambda d: float((d.loc[d["pred"] >= d["pred"].quantile(0.8), "target"] >= 0.3).mean())
    )  # actual top 20% = centred rank >= 0.3
    top_beat = g.apply(
        lambda d: float((d.loc[d["pred"] >= d["pred"].quantile(0.8), "target"] > 0).mean())
    )
    up_acc = g.apply(lambda d: float(((d["expected_return"] > 0) == (d["fwd_return"] > 0)).mean()))
    always_up = g.apply(lambda d: float((d["fwd_return"] > 0).mean()))
    ic = daily_ic(scored["pred"], scored["target"])
    spread = g.apply(
        lambda d: (
            d.loc[d["decile"] == 9, "fwd_return"].mean()
            - d.loc[d["decile"] == 0, "fwd_return"].mean()
        )
    )

    rows = [
        ("Beats-the-median accuracy", *t_vs(beat_acc, 0.5), "50.0% (coin flip)"),
        ("Top-20% precision (lands in actual top 20%)", *t_vs(top_prec, 0.2), "20.0% (random)"),
        ("Top-20% picks that beat the median", *t_vs(top_beat, 0.5), "50.0% (random)"),
        (
            "Up/down direction accuracy",
            *t_vs(up_acc, float(always_up.mean())),
            f"{always_up.mean():.1%} ('always up')",
        ),
    ]
    print("=== 1. ACCURACY on 2025-2026, all stocks, frozen model ===")
    for name, mean, t, base in rows:
        verdict = "significant" if abs(t) >= 2 else "not significant"
        print(f"  {name:46s} {mean:6.1%}   baseline {base:22s} t {t:+.2f} ({verdict})")
    ic_m, ic_t = t_vs(ic, 0.0)
    sp_m, sp_t = t_vs(spread, 0.0)
    print(
        f"  {'Information coefficient (rank correlation)':46s} {ic_m:+.3f}   baseline 0"
        f"{'':21s} t {ic_t:+.2f}"
    )
    print(
        f"  {'Top-decile minus bottom-decile return / month':46s} {sp_m:+.2%}   baseline 0"
        f"{'':21s} t {sp_t:+.2f}"
    )
    q = scored.assign(q=scored["decile"] // 2).groupby("q")["fwd_return"].mean()
    print(
        "  Realised 21-day return by predicted quintile (1 = worst .. 5 = best): "
        + "  ".join(f"Q{i + 1} {v:+.2%}" for i, v in q.items())
    )

    # ---------------- 2. price predictions for chosen stocks ----------------
    print("\n=== 2. PRICE PREDICTIONS one month ahead (split-adjusted $, frozen model) ===")
    all_rows = []
    for tkr in args.tickers:
        inst = f"EQ:{tkr}"
        if inst not in split.columns:
            continue
        hist_ret = closes[inst].loc[:train_end].pct_change(HORIZON).dropna().mean()
        recs = []
        for d in test_dates:
            if (d, inst) not in scored.index:
                continue
            r = scored.loc[(d, inst)]
            p0 = float(split.loc[d, inst])
            p1 = float(split.iloc[idx.get_loc(d) + HORIZON][inst])
            recs.append(
                {
                    "ticker": tkr,
                    "date": d.date(),
                    "price": p0,
                    "model_pct": float((r["pred"] > scored.xs(d, level="date")["pred"]).mean()),
                    "predicted_price": p0 * (1 + r["expected_return"]),
                    "naive_price": p0,  # random walk: next month = today
                    "drift_price": p0 * (1 + hist_ret),
                    "actual_price": p1,
                    "actual_return": p1 / p0 - 1,
                    "predicted_return": float(r["expected_return"]),
                }
            )
        df = pd.DataFrame(recs)
        if df.empty:
            continue
        all_rows.append(df)
        err = {
            k: (df[k] - df["actual_price"]).abs() / df["actual_price"]
            for k in ("predicted_price", "naive_price", "drift_price")
        }
        dir_acc = float(((df["predicted_return"] > 0) == (df["actual_return"] > 0)).mean())
        print(
            f"\n{tkr}: {len(df)} monthly predictions | mean abs % error: model "
            f"{err['predicted_price'].mean():.2%} | 'no change' {err['naive_price'].mean():.2%}"
            f" | historical drift {err['drift_price'].mean():.2%} | direction right "
            f"{dir_acc:.0%} (stock rose in {float((df['actual_return'] > 0).mean()):.0%})"
        )
        show = df[["date", "price", "model_pct", "predicted_price", "actual_price"]].copy()
        show["error"] = (show["predicted_price"] / show["actual_price"] - 1).map("{:+.1%}".format)
        show["model_pct"] = show["model_pct"].map("{:.0%}".format)
        for c in ("price", "predicted_price", "actual_price"):
            show[c] = show[c].map("${:,.2f}".format)
        print(show.rename(columns={"model_pct": "model rank"}).to_string(index=False))

    if not all_rows:
        raise SystemExit(f"none of {args.tickers} were in the test universe")
    pd.concat(all_rows).to_csv(OUT / "price_predictions.csv", index=False)
    pd.DataFrame(
        {
            "beat_median_acc": beat_acc,
            "top20_precision": top_prec,
            "ic": ic,
            "decile_spread": spread,
        }
    ).to_csv(OUT / "monthly_accuracy.csv")
    print(
        f"\nCSV written to {OUT}/ | calibration (training data): expected 21-day return by "
        "decile " + " ".join(f"{v:+.1%}" for v in decile_ret)
    )
    _ = stats


if __name__ == "__main__":
    main()
