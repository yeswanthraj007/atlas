"""One validation report for any strategy: walk-forward, signal scoring and scorecards.

Signal scoring is done per fold with the parameters that fold actually traded, so the
"signal" sections and the "trades" sections always describe the same decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from atlas.backtest.engine import performance_metrics
from atlas.validation import scorecard as sc
from atlas.validation.signal import (
    forward_returns,
    information_coefficient,
    per_ticker_signal_scores,
    summarize_ic,
)
from atlas.validation.stats import alpha_beta
from atlas.validation.walkforward import Params, WalkForwardResult

# (params) -> (ts, eligible) -> score per instrument at ts, higher = more attractive
SignalFactory = Callable[[Params], Callable[[pd.Timestamp, frozenset[str] | None], pd.Series]]


@dataclass
class OOSSignals:
    signals: pd.DataFrame  # rebalance date x instrument
    fwd: pd.DataFrame  # forward return over each fold's rebalance horizon
    top_n: pd.Series  # portfolio size in force on each date


def oos_signals(
    wf: WalkForwardResult,
    closes: pd.DataFrame,
    signal_for: SignalFactory,
    membership: pd.DataFrame | None,
) -> OOSSignals:
    sigs, fwds, tops = [], [], []
    for f in wf.folds:
        every = int(f.chosen.get("rebalance_every", 21))
        dates = f.test_equity.index[::every]
        fn = signal_for(f.chosen)
        rows = {}
        for ts in dates:
            elig = None
            if membership is not None:
                m = membership.loc[:ts]
                elig = frozenset(m.columns[m.iloc[-1].to_numpy(bool)]) if len(m) else None
            s = fn(ts, elig)
            if s is not None and len(s):
                rows[ts] = s
        if not rows:
            continue
        sig = pd.DataFrame(rows).T
        sigs.append(sig)
        fwds.append(forward_returns(closes, sig.index, every))
        tops.append(pd.Series(int(f.chosen.get("top_n", 20)), index=sig.index))
    return OOSSignals(
        pd.concat(sigs).sort_index(), pd.concat(fwds).sort_index(), pd.concat(tops).sort_index()
    )


def _pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.1%}"


def print_report(
    name: str,
    wf: WalkForwardResult,
    closes: pd.DataFrame,
    benchmark_close: pd.Series,
    signal_for: SignalFactory,
    membership: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    factor_ic: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> dict[str, float]:
    oos_days = wf.oos_returns.index
    bench_r = benchmark_close.pct_change().reindex(oos_days).fillna(0.0)
    oos_eq = wf.oos_equity(1_000_000)
    bench_eq = 1_000_000 * (1 + bench_r).cumprod()

    print(f"=== {name}: 1. Train vs test per fold (annualised Sharpe) ===")
    ft = wf.fold_table()
    print(
        ft.to_string(
            formatters={
                "train_sharpe": "{:.2f}".format,
                "test_sharpe": "{:.2f}".format,
                "test_return": _pct,
            }
        )
    )

    print(f"\n=== {name}: 2. Out-of-sample record vs benchmark (same days) ===")
    summ = wf.summary()
    bm = performance_metrics(bench_eq)
    rows = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"]
    tbl = pd.DataFrame(
        {"strategy": [summ[k] for k in rows], "benchmark": [bm[k] for k in rows]}, index=rows
    )
    excess = wf.oos_returns - bench_r
    te = float(excess.std(ddof=1) * (252**0.5))
    info_ratio = float(excess.mean() * 252 / te) if te else 0.0
    shown = tbl.astype(object)
    for k in rows:
        f = "{:.2f}" if k == "sharpe" else "{:.1%}"
        shown.loc[k] = [f.format(v) for v in tbl.loc[k]]
    print(shown.to_string())
    print(
        f"excess return vs benchmark {excess.mean() * 252:.1%}/yr | tracking error {te:.1%} | "
        f"information ratio {info_ratio:.2f}"
    )
    ab = alpha_beta(wf.oos_returns.to_numpy(), bench_r.to_numpy())
    print(
        f"vs benchmark: beta {ab['beta']:.2f} | ALPHA {ab['alpha_ann']:+.1%}/yr "
        f"(t {ab['alpha_t']:+.2f}) <- skill; |t| < 2 means no evidence of skill"
    )
    print(
        f"train Sharpe {summ['mean_train_sharpe']:.2f} -> test Sharpe "
        f"{summ['mean_test_sharpe']:.2f} (ratio {summ['oos_to_is_sharpe_ratio']:.2f}) | "
        f"profitable test periods {summ['profitable_folds']:.0%} | deflated Sharpe "
        f"P(real) {summ['deflated_sharpe_prob']:.1%} after {wf.n_trials} trials"
    )

    sig = oos_signals(wf, closes, signal_for, membership)
    s = summarize_ic(information_coefficient(sig.signals, sig.fwd))
    print(f"\n=== {name}: 3. Signal quality (IC of the traded signal, per-fold settings) ===")
    print(
        f"dates {s.n_dates} | mean IC {s.mean_ic:.3f} | t-stat {s.t_stat:.2f} | "
        f"IC>0 on {s.hit_rate:.0%} | IC IR {s.ic_ir:.2f}"
    )
    if factor_ic is not None:
        print("Each factor on its own (out-of-sample dates, non-overlapping):")
        fic = factor_ic.reindex(sig.signals.index).dropna(how="all")
        rows_f = {c: summarize_ic(fic[c]) for c in fic.columns if fic[c].notna().sum() > 2}
        print(
            pd.DataFrame(
                {
                    k: {
                        "mean_ic": f"{v.mean_ic:.3f}",
                        "t_stat": f"{v.t_stat:.2f}",
                        "ic_positive": f"{v.hit_rate:.0%}",
                    }
                    for k, v in rows_f.items()
                }
            ).T.to_string()
        )

    print(f"\n=== {name}: 4. Year by year ===")
    yt = sc.yearly_table(oos_eq, bench_eq)
    print(yt.to_string(formatters={c: _pct for c in yt.columns}))

    print(f"\n=== {name}: 5. Market regimes ===")
    rt = sc.regime_table(wf.oos_returns, benchmark_close)
    print(
        rt.to_string(
            formatters={
                "days": "{:.0f}".format,
                "ann_return": _pct,
                "ann_vol": _pct,
                "sharpe": "{:.2f}".format,
                "benchmark_ann_return": _pct,
            }
        )
    )

    print(f"\n=== {name}: 6. Worst drawdowns ===")
    dd = sc.drawdown_table(oos_eq)
    for col in ["start", "trough", "recovered"]:
        dd[col] = pd.to_datetime(dd[col]).dt.date
    print(dd.to_string(formatters={"depth": _pct}))

    trips = pd.concat([f.trips for f in wf.folds], ignore_index=True)
    card = sc.ticker_scorecard(trips)
    card.index = card.index.str.split(":").str[-1]
    fmt = {
        "total_pnl": "${:,.0f}".format,
        "win_rate": _pct,
        "avg_return": _pct,
        "best_return": _pct,
        "worst_return": _pct,
        "avg_days_held": "{:.0f}".format,
        "pnl_share": _pct,
    }
    print(f"\n=== {name}: 7. Per-ticker trades ({len(card)} tickers) ===")
    print("Best 8:\n" + card.head(8).to_string(formatters=fmt))
    print("Worst 8:\n" + card.tail(8).to_string(formatters=fmt))

    per = per_ticker_signal_scores(sig.signals, sig.fwd, top_n=sig.top_n)
    per.index = per.index.str.split(":").str[-1]
    if tickers:
        chosen = [t for t in tickers if t in per.index]
        print(f"\n=== {name}: 8. Selected tickers ===")
        view = per.loc[chosen].join(card[["trips", "total_pnl", "win_rate"]], how="left")
        print(
            view.to_string(
                na_rep="-",
                formatters={
                    "rank_ic": "{:.2f}".format,
                    "hit_rate_when_selected": _pct,
                    "avg_excess_when_selected": _pct,
                    "total_pnl": "${:,.0f}".format,
                    "win_rate": _pct,
                },
            )
        )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ft.to_csv(out_dir / "folds.csv")
        card.to_csv(out_dir / "ticker_trades.csv")
        per.to_csv(out_dir / "ticker_signal_scores.csv")
        yt.to_csv(out_dir / "yearly.csv")
        rt.to_csv(out_dir / "regimes.csv")
        pd.DataFrame({"strategy": oos_eq, "benchmark": bench_eq}).to_csv(out_dir / "oos_equity.csv")
    return {**summ, **ab, "information_ratio": info_ratio, "mean_ic": s.mean_ic, "ic_t": s.t_stat}
