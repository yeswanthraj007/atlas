"""Walk-forward validation report for a strategy on the S&P 500.

    uv run python scripts/validate.py --strategy multifactor
    uv run python scripts/validate.py --strategy momentum --tickers AAPL NVDA TSLA COST

The most recent year is a LOCKED HOLDOUT and is never evaluated here.
"""

from __future__ import annotations

import argparse
import itertools
import time
from decimal import Decimal as D
from pathlib import Path

import pandas as pd

from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.execution.broker import CostModel
from atlas.factors.panel import FactorPanel
from atlas.marketdata.loader import close_panel, load_for_backtest
from atlas.marketdata.store import BarStore
from atlas.marketdata.universe import membership_from_additions, sp500_addition_dates
from atlas.oms.risk_checks import MaxGrossExposure, MaxPositionWeight, NoShortSelling
from atlas.refdata.instruments import SecurityMaster
from atlas.strategy.momentum import CrossSectionalMomentum
from atlas.strategy.multifactor import MultiFactor
from atlas.validation.report import print_report
from atlas.validation.splits import walk_forward_splits
from atlas.validation.walkforward import walk_forward

CONSTITUENTS = "data/universes/sp500_constituents.csv"
COSTS = CostModel(per_share=D(0), min_per_order=D(0), bps_of_notional=D(1), slippage_bps=D(5))


def config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=D(1_000_000),
        costs=COSTS,
        max_participation=0.01,
        rules=lambda: [MaxPositionWeight(D("0.10")), MaxGrossExposure(D("1.00")), NoShortSelling()],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["momentum", "multifactor"], default="multifactor")
    ap.add_argument(
        "--tickers",
        nargs="*",
        default=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "COST", "JPM", "XOM"],
    )
    args = ap.parse_args()

    store = BarStore("data")
    sm = SecurityMaster()
    bars = load_for_backtest(store, Path("data/universes/sp500.txt").read_text().split(), sm)
    spy_close = store.read("SPY")["close"]
    closes = close_panel(bars)
    members = membership_from_additions(sp500_addition_dates(CONSTITUENTS), closes.index)
    sectors_raw = pd.read_csv(CONSTITUENTS)
    sectors = dict(
        zip(
            "EQ:" + sectors_raw["Symbol"].str.replace("-", ".", regex=False),
            sectors_raw["GICS Sector"],
            strict=True,
        )
    )
    plan = walk_forward_splits(
        closes.index,
        closes.index[0] + pd.Timedelta(days=380),
        min_train_days=756,
        test_days=252,
        embargo_days=21,
        holdout_days=252,
    )

    if args.strategy == "momentum":
        grid = [
            {"lookback": lb, "skip": 21, "top_n": n, "rebalance_every": 21, "max_weight": 0.08}
            for lb, n in itertools.product([126, 252], [10, 20, 40])
        ]

        def make(p):
            return CrossSectionalMomentum(**p)

        def signal_for(p):
            strat = CrossSectionalMomentum(**p)
            return lambda ts, elig: strat.signal(
                closes.loc[:ts, [c for c in closes.columns if elig is None or c in elig]]
            )

        factor_ic = None
    else:
        t0 = time.time()
        panel = FactorPanel.from_closes(closes, horizon=21, membership=members)
        factor_ic = panel.daily_ic()
        print(
            f"factor panel: {len(panel.factors)} factors x {closes.shape} in "
            f"{time.time() - t0:.1f}s"
        )
        grid = [
            {"weighting": w, "top_n": n, "trend_filter": tf, "rebalance_every": 21}
            for w, n, tf in itertools.product(["ic", "equal"], [20, 40], [False, True])
        ]

        def make(p):
            return MultiFactor(panel, sectors=sectors, benchmark=spy_close, **p)

        def signal_for(p):
            strat = make(p)
            return lambda ts, elig: strat.scores(ts, elig)

    print(
        f"{len(bars)} stocks | {len(plan.folds)} folds | {len(grid)} parameter sets | "
        f"holdout {plan.holdout_start.date()}..{plan.holdout_end.date()} (locked)"
    )
    t0 = time.time()
    wf = walk_forward(
        make, grid, bars, sm, plan, config, membership=members, warmup_days=400, progress=print
    )
    print(f"walk-forward done in {time.time() - t0:.0f}s\n")
    print_report(
        args.strategy,
        wf,
        closes,
        spy_close,
        signal_for,
        members,
        args.tickers,
        factor_ic,
        Path(f"data/reports/validation/{args.strategy}"),
    )
    print(
        "\nCAVEAT: stocks later removed from the S&P 500 are missing (residual "
        "survivorship bias), so results are somewhat optimistic."
    )
    _ = run_backtest  # re-exported for interactive use


if __name__ == "__main__":
    main()
