"""Walk-forward validation of S&P 500 momentum, with scorecards.

    uv run python scripts/validate_momentum.py [--tickers AAPL MSFT NVDA ...]

The most recent year is a LOCKED HOLDOUT and is not evaluated here. Run
`--final-holdout` once, only after all strategy development is finished.
"""

from __future__ import annotations

import argparse
import itertools
import time
from decimal import Decimal as D
from pathlib import Path

import pandas as pd

from atlas.backtest.engine import BacktestConfig, performance_metrics, run_backtest
from atlas.execution.broker import CostModel
from atlas.marketdata.loader import close_panel, load_for_backtest
from atlas.marketdata.store import BarStore
from atlas.marketdata.universe import membership_from_additions, sp500_addition_dates
from atlas.oms.risk_checks import MaxGrossExposure, MaxPositionWeight, NoShortSelling
from atlas.refdata.instruments import SecurityMaster
from atlas.strategy.base import StrategyContext
from atlas.strategy.momentum import CrossSectionalMomentum
from atlas.validation import scorecard as sc
from atlas.validation.signal import (
    forward_returns,
    information_coefficient,
    per_ticker_signal_scores,
    signal_panel,
    summarize_ic,
)
from atlas.validation.splits import walk_forward_splits
from atlas.validation.walkforward import evaluate_holdout, walk_forward

OUT = Path("data/reports/validation")
COSTS = CostModel(per_share=D(0), min_per_order=D(0), bps_of_notional=D(1), slippage_bps=D(5))
GRID = [
    {"lookback": lb, "skip": 21, "top_n": n, "rebalance_every": 21, "max_weight": 0.08}
    for lb, n in itertools.product([126, 252], [10, 20, 40])
]


def config(max_position: str = "0.10") -> BacktestConfig:
    return BacktestConfig(
        initial_cash=D(1_000_000),
        costs=COSTS,
        max_participation=0.01,
        rules=lambda: [
            MaxPositionWeight(D(max_position)),
            MaxGrossExposure(D("1.00")),
            NoShortSelling(),
        ],
    )


class BuyAndHold:
    id = "buy_and_hold"

    def on_bar(self, ctx: StrategyContext) -> dict[str, float] | None:
        return {"ETF:SPY": 0.98} if ctx.bar_index == 0 else None


def pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tickers",
        nargs="*",
        default=["AAPL", "MSFT", "NVDA", "AMZN", "META", "JPM", "XOM", "JNJ", "TSLA", "WMT"],
    )
    ap.add_argument("--final-holdout", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    store = BarStore("data")
    sm = SecurityMaster()
    stocks = Path("data/universes/sp500.txt").read_text().split()
    bars = load_for_backtest(store, stocks, sm)
    spy_sm = SecurityMaster()
    spy = load_for_backtest(store, ["SPY"], spy_sm, etfs={"SPY"})
    closes = close_panel(bars)
    members = membership_from_additions(
        sp500_addition_dates("data/universes/sp500_constituents.csv"), closes.index
    )
    plan = walk_forward_splits(
        closes.index,
        closes.index[0] + pd.Timedelta(days=380),
        min_train_days=756,
        test_days=252,
        embargo_days=21,
        holdout_days=252,
    )
    print(
        f"{len(bars)} stocks | {len(plan.folds)} folds | {len(GRID)} parameter sets | "
        f"holdout {plan.holdout_start.date()}..{plan.holdout_end.date()} (locked)"
    )

    t0 = time.time()
    wf = walk_forward(
        lambda p: CrossSectionalMomentum(**p),
        GRID,
        bars,
        sm,
        plan,
        config,
        membership=members,
        warmup_days=400,
        progress=print,
    )
    print(f"walk-forward done in {time.time() - t0:.0f}s\n")

    # Benchmarks on exactly the out-of-sample days
    oos_days = wf.oos_returns.index
    spy_eq = run_backtest(BuyAndHold(), spy, spy_sm, config("1.00")).equity
    spy_r = spy_eq.pct_change().reindex(oos_days)
    oos_eq = wf.oos_equity(1_000_000)
    spy_oos_eq = 1_000_000 * (1 + spy_r.fillna(0)).cumprod()

    print("=== 1. Train vs test, per fold (Sharpe is annualised) ===")
    ft = wf.fold_table()
    print(
        ft.to_string(
            formatters={
                "train_sharpe": "{:.2f}".format,
                "test_sharpe": "{:.2f}".format,
                "test_return": pct,
            }
        )
    )

    print("\n=== 2. Out-of-sample record vs SPY (same days) ===")
    summ = wf.summary()
    spy_m = performance_metrics(spy_oos_eq)
    rows = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"]
    tbl = pd.DataFrame(
        {"momentum_oos": [summ[k] for k in rows], "spy": [spy_m[k] for k in rows]}, index=rows
    )
    shown = tbl.copy().astype(object)
    for k in rows:
        f = "{:.2f}" if k == "sharpe" else "{:.1%}"
        shown.loc[k] = [f.format(v) for v in tbl.loc[k]]
    print(shown.to_string())
    print(
        f"mean train Sharpe {summ['mean_train_sharpe']:.2f} -> mean test Sharpe "
        f"{summ['mean_test_sharpe']:.2f} (ratio {summ['oos_to_is_sharpe_ratio']:.2f})"
    )
    print(f"profitable test years: {summ['profitable_folds']:.0%}")
    print(
        f"deflated Sharpe: P(skill is real after {wf.n_trials} trials) = "
        f"{summ['deflated_sharpe_prob']:.1%}"
    )

    print("\n=== 3. Signal quality (information coefficient, 21-day horizon) ===")
    strat = CrossSectionalMomentum(lookback=252, skip=21)
    rebal = oos_days[::21]
    sigs = signal_panel(closes, strat.signal, rebal, eligible=members)
    fwd = forward_returns(closes, rebal, 21)
    ic = information_coefficient(sigs, fwd)
    s = summarize_ic(ic)
    print(
        f"dates {s.n_dates} | mean IC {s.mean_ic:.3f} | t-stat {s.t_stat:.2f} | "
        f"IC>0 on {s.hit_rate:.0%} of dates | IC IR {s.ic_ir:.2f}"
    )

    print("\n=== 4. Year by year (out of sample) ===")
    yt = sc.yearly_table(oos_eq, spy_oos_eq)
    print(yt.to_string(formatters={c: pct for c in yt.columns}))

    print("\n=== 5. Market regimes: where it works, where it fails ===")
    rt = sc.regime_table(wf.oos_returns, spy["ETF:SPY"]["close"])
    print(
        rt.to_string(
            formatters={
                "days": "{:.0f}".format,
                "ann_return": pct,
                "ann_vol": pct,
                "sharpe": "{:.2f}".format,
                "benchmark_ann_return": pct,
            }
        )
    )

    print("\n=== 6. Worst drawdowns (out of sample) ===")
    print(sc.drawdown_table(oos_eq).to_string(formatters={"depth": pct}))

    trips = pd.concat([f.trips for f in wf.folds], ignore_index=True)
    card = sc.ticker_scorecard(trips)
    card.index = card.index.str.replace("EQ:", "")
    fmt = {
        "total_pnl": "${:,.0f}".format,
        "win_rate": pct,
        "avg_return": pct,
        "best_return": pct,
        "worst_return": pct,
        "avg_days_held": "{:.0f}".format,
        "pnl_share": pct,
    }
    print(f"\n=== 7. Per-ticker trade scorecard ({len(card)} tickers traded) ===")
    print("Best 10:\n" + card.head(10).to_string(formatters=fmt))
    print("Worst 10:\n" + card.tail(10).to_string(formatters=fmt))

    per_tkr = per_ticker_signal_scores(sigs, fwd, top_n=20)
    per_tkr.index = per_tkr.index.str.replace("EQ:", "")
    chosen = [t for t in args.tickers if t in per_tkr.index]
    print("\n=== 8. Selected tickers: signal score and trading results ===")
    view = per_tkr.loc[chosen].join(card[["trips", "total_pnl", "win_rate"]], how="left")
    print(
        view.to_string(
            formatters={
                "rank_ic": "{:.2f}".format,
                "hit_rate_when_selected": pct,
                "avg_excess_when_selected": pct,
                "total_pnl": "${:,.0f}".format,
                "win_rate": pct,
            },
            na_rep="-",
        )
    )

    ft.to_csv(OUT / "folds.csv")
    card.to_csv(OUT / "ticker_scorecard.csv")
    per_tkr.to_csv(OUT / "ticker_signal_scores.csv")
    yt.to_csv(OUT / "yearly.csv")
    rt.to_csv(OUT / "regimes.csv")
    pd.DataFrame({"momentum_oos": oos_eq, "spy": spy_oos_eq}).to_csv(OUT / "oos_equity.csv")
    print(f"\nCSV reports written to {OUT}/")
    print(
        "CAVEAT: stocks later REMOVED from the S&P 500 are still missing (residual "
        "survivorship bias). Results remain somewhat optimistic."
    )

    if args.final_holdout:
        from collections import Counter

        params = dict(
            Counter(tuple(sorted(f.chosen.items())) for f in wf.folds).most_common(1)[0][0]
        )
        hold_eq, _ = evaluate_holdout(
            lambda p: CrossSectionalMomentum(**p),
            params,
            bars,
            sm,
            plan,
            config,
            membership=members,
        )
        print("\n=== FINAL HOLDOUT (evaluated once) ===", params)
        print({k: round(v, 3) for k, v in performance_metrics(hold_eq).items()})


if __name__ == "__main__":
    main()
