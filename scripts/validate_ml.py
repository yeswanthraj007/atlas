"""Machine-learning alpha model: walk-forward training, scoring and portfolio backtest.

    uv run python scripts/validate_ml.py --features all --models ridge gbm
    uv run python scripts/validate_ml.py --features price
    uv run python scripts/validate_ml.py --features fundamental

The most recent year is a LOCKED HOLDOUT and is never evaluated here.
"""

from __future__ import annotations

import argparse
import time
from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.backtest.engine import BacktestConfig
from atlas.execution.broker import CostModel
from atlas.factors.fundamental import FUNDAMENTAL_FACTORS, fundamental_factor_frames
from atlas.factors.panel import price_factor_frames
from atlas.marketdata.bars import Adjustment
from atlas.marketdata.loader import close_panel, load_for_backtest
from atlas.marketdata.sec import items_for, load_facts
from atlas.marketdata.store import BarStore
from atlas.marketdata.universe import membership_from_additions, sp500_addition_dates
from atlas.ml.dataset import build_dataset, feature_columns
from atlas.ml.models import GBMRanker, RidgeRanker
from atlas.ml.walkforward import ml_walk_forward, sample_dates
from atlas.oms.risk_checks import MaxGrossExposure, MaxPositionWeight, NoShortSelling
from atlas.refdata.instruments import SecurityMaster
from atlas.validation.report import print_report
from atlas.validation.signal import summarize_ic
from atlas.validation.splits import walk_forward_splits

CONSTITUENTS = "data/universes/sp500_constituents.csv"
COSTS = CostModel(per_share=D(0), min_per_order=D(0), bps_of_notional=D(1), slippage_bps=D(5))

MODELS = {
    "ridge": lambda: RidgeRanker(alpha=10.0),
    "gbm_shallow": lambda: GBMRanker(max_leaf_nodes=7, min_samples_leaf=800),
    "gbm": lambda: GBMRanker(max_leaf_nodes=15, min_samples_leaf=400),
}


def config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=D(1_000_000),
        costs=COSTS,
        max_participation=0.01,
        rules=lambda: [MaxPositionWeight(D("0.10")), MaxGrossExposure(D("1.00")), NoShortSelling()],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["price", "fundamental", "all"], default="all")
    ap.add_argument("--models", nargs="*", default=["ridge", "gbm_shallow", "gbm"])
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "NVDA", "TSLA", "COST", "JPM"])
    args = ap.parse_args()

    t0 = time.time()
    store = BarStore("data")
    sm = SecurityMaster()
    symbols = Path("data/universes/sp500.txt").read_text().split()
    bars = load_for_backtest(store, symbols, sm)
    closes = close_panel(bars)
    members = membership_from_additions(sp500_addition_dates(CONSTITUENTS), closes.index)
    spy = store.read("SPY")["close"]
    sec = pd.read_csv(CONSTITUENTS)
    sectors = dict(
        zip(
            "EQ:" + sec["Symbol"].str.replace("-", ".", regex=False),
            sec["GICS Sector"],
            strict=True,
        )
    )

    factors: dict[str, pd.DataFrame] = {}
    if args.features in ("price", "all"):
        factors.update(price_factor_frames(closes))
    if args.features in ("fundamental", "all"):
        split = pd.DataFrame(
            {
                f"EQ:{s}": store.read(s, adjustment=Adjustment.SPLIT)["close"]
                for s in symbols
                if f"EQ:{s}" in closes.columns
            }
        )
        raw = pd.DataFrame(
            {
                f"EQ:{s}": store.read(s, adjustment=Adjustment.RAW)["close"]
                for s in symbols
                if f"EQ:{s}" in closes.columns
            }
        )
        split, raw = split.reindex(closes.index), raw.reindex(closes.index)
        fund = fundamental_factor_frames(items_for(load_facts(), symbols), split, raw)
        factors.update({k: fund[k] for k in FUNDAMENTAL_FACTORS if k in fund})
        cover = fund["earnings_yield"].notna().sum(axis=1)
        print(
            f"fundamentals: {len(fund) - 1} factors, coverage "
            f"{int(cover.loc['2018':].median())} stocks/day (median)"
        )

    plan = walk_forward_splits(
        closes.index,
        closes.index[0] + pd.Timedelta(days=380),
        min_train_days=756,
        test_days=252,
        embargo_days=21,
        holdout_days=252,
    )
    data = build_dataset(factors, closes, sample_dates(closes.index, plan), 21, members, spy)
    print(
        f"dataset: {len(data):,} rows x {len(feature_columns(data))} features "
        f"({time.time() - t0:.0f}s)"
    )

    configs = {k: MODELS[k] for k in args.models}
    t0 = time.time()
    res = ml_walk_forward(
        data,
        configs,
        plan,
        bars,
        closes,
        sm,
        config,
        members,
        sectors,
        top_n=args.top_n,
        progress=print,
    )
    print(f"ML walk-forward done in {time.time() - t0:.0f}s\n")

    s = summarize_ic(res.oos_ic)
    print("=== Model selection (purged CV IC on training data, per fold) ===")
    print(res.cv_table.round(4).to_string())
    print(
        f"\nOut-of-sample prediction IC: mean {s.mean_ic:.4f} | t-stat {s.t_stat:.2f} | "
        f"IC>0 on {s.hit_rate:.0%} of {s.n_dates} dates"
    )
    imp = res.importance.mean().sort_values(ascending=False)
    print("\n=== What the model relies on (mean importance across folds) ===")
    print(imp.head(12).round(5).to_string())

    pred = res.predictions

    def signal_for(_p):
        def fn(ts: pd.Timestamp, elig: frozenset[str] | None) -> pd.Series:
            if ts not in pred.index:
                return pd.Series(dtype=float)
            row = pred.loc[ts].dropna()
            return row[row.index.isin(elig)] if elig is not None else row

        return fn

    print()
    print_report(
        f"ml[{args.features}]",
        res.wf,
        closes,
        spy,
        signal_for,
        members,
        args.tickers,
        None,
        Path(f"data/reports/validation/ml_{args.features}"),
    )
    out = Path(f"data/reports/validation/ml_{args.features}")
    res.cv_table.to_csv(out / "cv_ic.csv")
    res.importance.to_csv(out / "importance.csv")
    res.oos_ic.to_csv(out / "oos_ic.csv")
    print(
        "\nCAVEAT: stocks later removed from the S&P 500 are missing (residual "
        "survivorship bias), so results are somewhat optimistic."
    )
    _ = np


if __name__ == "__main__":
    main()
