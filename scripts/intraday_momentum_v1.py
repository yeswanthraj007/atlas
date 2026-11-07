# ruff: noqa: E501
"""Market intraday momentum v1, exactly as pre-registered in
research/preregistered/2026-11-07-intraday-momentum-v1.md.

    uv run python scripts/intraday_momentum_v1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atlas.factors.intraday import daily_pnl, session_table, summarize
from atlas.lab.trials import log_external_trial
from atlas.recommend.engine import Registry, SignalRecord

PREREG = "research/preregistered/2026-11-07-intraday-momentum-v1.md"
DEV_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
COSTS = (1.0, 0.5, 2.0)  # round-trip bp: primary first


def load(sym: str) -> pd.DataFrame:
    parts = sorted(Path("data/minute", sym).glob("*.parquet"))
    return session_table(pd.concat([pd.read_parquet(p) for p in parts]).sort_index())


def positions(t: pd.DataFrame, beta: np.ndarray, cutoff: float) -> dict[str, pd.Series]:
    a = np.sign(t["r1"]).astype(int)
    pred = beta[0] + beta[1] * t["r1"] + beta[2] * t["r12"]
    b = np.sign(pred).astype(int)
    c = a.where(t["vol20"] >= cutoff, 0)  # vol20 NaN compares False: flat
    return {"A": a, "B": b, "C": c}


def fmt(s: dict) -> str:
    return (f"{s['trades']:>5d} trades | {s['mean_net_bp']:+6.2f} bp/trade (t {s['t']:+5.2f}) | "
            f"hit {s['hit_rate']:.1%} | Sharpe {s['sharpe']:+5.2f} | total {s['total_net_pct']:+6.1f}% "
            f"| max DD {s['max_drawdown_pct']:.1f}%")  # fmt: skip


def main() -> None:
    print(f"pre-registration: {PREREG}")
    results: dict[str, dict] = {}
    for sym in ("SPY", "QQQ", "IWM"):
        t = load(sym)
        dev, test = t[t.index <= DEV_END], t[t.index >= TEST_START]
        x = np.column_stack([np.ones(len(dev)), dev["r1"], dev["r12"]])
        beta = np.linalg.lstsq(x, dev["r13"].to_numpy(), rcond=None)[0]
        cutoff = float(dev["vol20"].quantile(2 / 3))
        print(
            f"\n===== {sym}: {len(t):,} sessions ({len(dev):,} development, {len(test):,} frozen test) ====="
        )
        print(f"development fit: r13 = {beta[0] * 1e4:+.2f} bp + {beta[1]:+.4f} r1 + {beta[2]:+.4f} r12 "
              f"| volatility cut-off (67th pct) {cutoff * 100:.3f}%/day")  # fmt: skip
        res = {}
        for name, df in (("development", dev), ("FROZEN TEST", test)):
            pos = positions(df, beta, cutoff)
            print(f"-- {name} (cost 1.0 bp round trip)")
            for k in "ABC":
                s = summarize(daily_pnl(pos[k], df["trade_ret"], 1.0), pos[k])
                res[(name, k)] = s
                print(f"   {k}: {fmt(s)}")
        if sym == "SPY":
            pos = positions(test, beta, cutoff)
            print("-- FROZEN TEST, strategy A, cost sensitivity")
            for c in COSTS[1:]:
                print(
                    f"   {c:.1f} bp: {fmt(summarize(daily_pnl(pos['A'], test['trade_ret'], c), pos['A']))}"
                )
            print("-- FROZEN TEST, strategy A by year (1.0 bp)")
            for y, g in test.groupby(test.index.year):
                pa = positions(g, beta, cutoff)["A"]
                print(f"   {y}: {fmt(summarize(daily_pnl(pa, g['trade_ret'], 1.0), pa))}")
            hi = test["vol20"] >= cutoff
            ra = daily_pnl(pos["A"], test["trade_ret"], 1.0)
            w = stats.ttest_ind(ra[hi], ra[~hi], equal_var=False)
            print(f"-- volatility interaction, frozen test: A on high-vol days {ra[hi].mean() * 1e4:+.2f} bp "
                  f"({int(hi.sum())} days) vs low/mid-vol days {ra[~hi].mean() * 1e4:+.2f} bp "
                  f"({int((~hi).sum())} days); Welch t {w.statistic:+.2f}")  # fmt: skip
            results["interaction_t"] = float(w.statistic)
        results[sym] = res

    spy = results["SPY"]
    a = spy[("FROZEN TEST", "A")]
    ok = [a["mean_net_bp"] > 0 and a["t"] >= 3.0, a["hit_rate"] > 0.51, a["sharpe"] >= 0.5]
    print("\n=== Success criteria, strategy A on SPY, frozen test ===")
    for label, passed in zip(
        ["mean net > 0 with t >= 3.0", "hit rate > 51%", "net Sharpe >= 0.5"], ok, strict=True
    ):
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    promoted = all(ok)
    print(
        f"VERDICT: {'PROMOTED to paper trading as an execution candidate' if promoted else 'rejected'}"
    )
    for k in "ABC":
        s = spy[("FROZEN TEST", k)]
        metrics = {**s, "development": spy[("development", k)],
                   "qqq_test": results["QQQ"][("FROZEN TEST", k)], "iwm_test": results["IWM"][("FROZEN TEST", k)]}  # fmt: skip
        if k == "C":
            metrics["interaction_welch_t"] = results["interaction_t"]
        t = log_external_trial(
            f"intraday_momentum_{k.lower()}", "promoted" if (k == "A" and promoted) else "rejected_holdout",
            f"intraday momentum strategy {k} on SPY earns a net profit in the last half-hour",
            "Gao, Han, Li & Zhou (2018) out-of-sample 2016-2026", metrics, True, __file__,
            notes=f"pre-registered in {PREREG}")  # fmt: skip
        print(f"  trial #{t.trial_id}: strategy {k}")
        if k == "A" and promoted:
            reg = Registry.load()
            reg.record(SignalRecord("intraday_momentum_spy", "SPY first-half-hour to last-half-hour",
                                    "direction", "net_bp_per_trade_t", s["mean_net_bp"], s["t"],
                                    "frozen 2023-2026", f"trial #{t.trial_id}"))  # fmt: skip
            reg.save()


if __name__ == "__main__":
    main()
