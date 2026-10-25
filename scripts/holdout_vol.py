"""Frozen volatility model: train on 2016-2024, forecast every month of 2025-2026.

uv run python scripts/holdout_vol.py --tickers AAPL NVDA TSLA COST JPM XOM
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.factors.earnings import earnings_factor_frames
from atlas.factors.fundamental import fundamental_factor_frames
from atlas.marketdata.bars import Adjustment
from atlas.marketdata.sec import items_for, load_facts
from atlas.marketdata.store import BarStore
from atlas.ml.models import GBMRanker, RidgeRanker, cross_validate
from atlas.ml.volatility import HAR_FEATURES, build_vol_dataset, evaluate_vol, vol_feature_columns
from atlas.research import load_research_data

HORIZON = 21
OUT = Path("data/reports/holdout_vol")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "NVDA", "TSLA", "COST", "JPM", "XOM"])
    args = ap.parse_args()
    args.tickers = [t.upper() for x in args.tickers for t in re.split(r"[\s,]+", x) if t]
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    OUT.mkdir(parents=True, exist_ok=True)

    rd = load_research_data("price")
    closes, idx = rd.closes, rd.closes.index
    store = BarStore("data")
    symbols = [c.split(":")[1] for c in closes.columns]
    facts = items_for(load_facts(), symbols)
    raw = pd.DataFrame(
        {f"EQ:{s}": store.read(s, adjustment=Adjustment.RAW)["close"] for s in symbols}
    ).reindex(idx)
    mcap = fundamental_factor_frames(facts, rd.split_closes, raw)["market_cap"]
    earn = earnings_factor_frames(facts, idx)
    extra = {"log_mcap": np.log(mcap), "days_since_filing": earn["days_since_filing"]}

    train_grid = idx[(idx >= idx[0] + pd.Timedelta(days=380)) & (idx <= train_end)][::5]
    after = idx[idx > train_end]
    test_dates = after[::HORIZON]
    test_dates = test_dates[idx.get_indexer(test_dates) + HORIZON < len(idx)]
    data = build_vol_dataset(
        closes, train_grid.union(test_dates), HORIZON, extra, rd.benchmark, rd.membership
    )
    feats = vol_feature_columns(data)
    label_end = pd.to_datetime(data["label_end"])
    train = data[(label_end <= train_end) & data["target"].notna()]
    test = data[data.index.get_level_values("date").isin(test_dates) & data["target"].notna()]
    print(
        f"train {len(train):,} rows to {train_end.date()} | test {len(test):,} predictions, "
        f"{len(test_dates)} months, model FROZEN | features: {feats}\n"
    )

    candidates = {
        "har": (lambda: RidgeRanker(alpha=1e-6), HAR_FEATURES),
        "ridge_all": (lambda: RidgeRanker(alpha=1.0), feats),
        "gbm_all": (
            lambda: GBMRanker(
                max_leaf_nodes=31, min_samples_leaf=200, max_iter=600, learning_rate=0.05
            ),
            feats,
        ),
    }
    cv = {k: cross_validate(make, train, cols) for k, (make, cols) in candidates.items()}
    best = max(cv, key=lambda k: cv[k])
    print(
        "model choice by purged CV (rank correlation of forecasts):",
        ", ".join(f"{k} {v:.3f}" for k, v in cv.items()),
        "->",
        best,
    )

    preds = {"naive (last month's vol)": test["lrv_21"]}
    for name, (make, cols) in candidates.items():
        m = make().fit(train[cols], train["target"])
        preds[name] = pd.Series(m.predict(test[cols]), index=test.index)
    rows = {name: evaluate_vol(p, test) for name, p in preds.items()}
    table = pd.DataFrame(rows).T
    print("\n=== Volatility forecast accuracy, 2025-2026, frozen models ===")
    fmt = {
        "r2_log": "{:.3f}",
        "mae_vol_points": "{:.2f}",
        "median_abs_pct_error": "{:.1%}",
        "acc_above_median": "{:.1%}",
        "acc_above_median_min_month": "{:.1%}",
        "acc_vol_up_or_down": "{:.1%}",
        "top20_precision": "{:.1%}",
    }
    print(table.apply(lambda col: col.map(fmt[col.name].format)).to_string())
    print(
        "\nacc_above_median: will this stock be more volatile than the typical stock next "
        "month? (coin flip = 50%)\nacc_vol_up_or_down: will its volatility rise or fall "
        "vs last month? (coin flip = 50%)\ntop20_precision: of the 20% predicted most "
        "volatile, how many were (random = 20%)"
    )

    best_pred = preds[best]
    print(f"\n=== Per-stock forecasts with {best} (annualised volatility) ===")
    out = []
    for tkr in args.tickers:
        inst = f"EQ:{tkr}"
        if inst not in test.index.get_level_values("instrument"):
            continue
        t = test.xs(inst, level="instrument")
        p = best_pred.xs(inst, level="instrument")
        df = pd.DataFrame(
            {
                "last_month": np.exp(t["lrv_21"]),
                "forecast": np.exp(p),
                "actual": np.exp(t["target"]),
            }
        )
        df["error"] = df["forecast"] / df["actual"] - 1
        err_m = (df["forecast"] / df["actual"] - 1).abs().mean()
        err_n = (df["last_month"] / df["actual"] - 1).abs().mean()
        print(
            f"{tkr}: mean abs error model {err_m:.1%} vs naive {err_n:.1%} | "
            f"actual vol range {df['actual'].min():.0%}-{df['actual'].max():.0%}"
        )
        out.append(df.assign(ticker=tkr))
    if not out:
        raise SystemExit(f"none of {args.tickers} were in the test universe")
    pd.concat(out).to_csv(OUT / "per_stock.csv")
    table.to_csv(OUT / "accuracy.csv")
    print(f"\nCSV written to {OUT}/")


if __name__ == "__main__":
    main()
