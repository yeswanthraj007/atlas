"""Earnings press-release signal (8-K Item 2.02), gated like the research lab.

    uv run python scripts/holdout_earnings.py

Hypothesis: the wording of a company's earnings press release, learned from how markets
reacted to past releases, predicts the drift in the days and weeks AFTER the release.

Gates
  1. development: train on releases through 2023, test on 2024 (unseen). Needs t >= 2 at
     the chosen horizon, else stop with the 2025-2026 holdout unopened.
  2. holdout: retrain through 2024, freeze, test 2025-2026. Needs t >= 2, same sign.
Each run is logged in research/trials.jsonl; a pass writes the signal registry.

Returns are market-adjusted (minus SPY). A release accepted after the 16:00 New York close
is first tradeable at the next open; the release-day move itself is never counted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atlas.lab.trials import log_external_trial
from atlas.marketdata.earnings_releases import load_earnings_index, read_release
from atlas.marketdata.store import BarStore
from atlas.ml.news_model import NewsSentimentModel, identity_tokens, news_day_index
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

OUT = Path("data/reports/holdout_earnings")
HORIZONS = {"next_day": 1, "next_5d": 5, "next_21d": 21, "next_63d": 63}
PRIMARY = "next_21d"
HYPOTHESIS = (
    "Earnings press releases whose wording resembles past releases that the market "
    "rewarded keep outperforming for weeks after the release"
)
RATIONALE = (
    "Post-earnings drift: investors underreact to earnings news (Bernard & Thomas "
    "1989). Text adds what headline EPS misses (guidance, tone, one-offs); BlackRock "
    "trains models on call text labelled by market reaction for the same reason"
)


def monthly_t(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """IC per calendar month (months are the independent unit) -> mean, t, accuracy."""
    ics = (
        df.groupby(df["month"])
        .apply(lambda g: stats.spearmanr(g["score"], g[col]).statistic if len(g) >= 15 else np.nan)
        .dropna()
    )
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 2 else np.nan
    acc = float((np.sign(df["score"]) == np.sign(df[col])).mean())
    return float(ics.mean()), float(t), acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=4000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rd = load_research_data("price")
    idx = rd.closes.index
    syms = [c.split(":")[1] for c in rd.closes.columns]
    col = {s: i for i, s in enumerate(syms)}
    c = rd.closes.set_axis(syms, axis=1).to_numpy()
    o = (
        pd.DataFrame({k.split(":")[1]: v["open"] for k, v in rd.bars.items()})
        .reindex(idx)
        .reindex(columns=syms)
        .to_numpy()
    )
    spy = BarStore("data").read("SPY").reindex(idx)
    sc, so = spy["close"].to_numpy(), spy["open"].to_numpy()

    ev = load_earnings_index()
    ev = ev[ev["symbol"].isin(col)].drop_duplicates("accession").copy()
    ev["text"] = [
        read_release("data/sec", r.cik, r.accession)[: args.max_chars].lower()
        for r in ev.itertuples()
    ]
    ev["d0"] = news_day_index(pd.to_datetime(ev["accepted"], utc=True), idx)
    ev = ev[(ev["d0"] >= 1) & (ev["d0"] + 63 < len(idx))].copy()
    d0, j = ev["d0"].to_numpy(), ev["symbol"].map(col).to_numpy()

    def abn(a: np.ndarray, b: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
        return a / b - sa / sb

    ev["reaction"] = abn(c[d0 + 1, j], c[d0 - 1, j], sc[d0 + 1], sc[d0 - 1])  # label only
    for name, h in HORIZONS.items():
        ev[name] = abn(c[d0 + h, j], o[d0 + 1, j], sc[d0 + h], so[d0 + 1])
    ev["date"] = idx[d0]
    ev["month"] = ev["date"].dt.strftime("%Y-%m")
    ev = ev.dropna(subset=["reaction", "next_day"])
    print(
        f"{len(ev):,} earnings releases with prices | "
        f"{ev['date'].min().date()} .. {ev['date'].max().date()}"
    )

    def run(train: pd.DataFrame, test: pd.DataFrame, label: str) -> dict[str, tuple]:
        masked = identity_tokens(train.rename(columns={"symbol": "symbol"}), min_count=20)
        model = NewsSentimentModel(stop_words=masked, min_df=10).fit(
            train["text"], train["reaction"]
        )
        test = test.assign(score=model.score(test["text"]))
        pos, neg = model.top_words(10)
        print(f"\n[{label}] train {len(train):,} releases -> test {len(test):,}")
        print("  learned positive:", ", ".join(pos))
        print("  learned negative:", ", ".join(neg))
        res = {}
        for name in HORIZONS:
            ic, t, acc = monthly_t(test, name)
            res[name] = (ic, t, acc)
            print(f"  {name:9s} IC {ic:+.3f} (t {t:+.2f}) | direction right {acc:.1%}")
        test.to_csv(OUT / f"{label}_scores.csv", index=False)
        return res

    y = ev["date"].dt.year
    dev = run(
        ev[(y <= 2023) & (ev["d0"] + 63 < idx.searchsorted(pd.Timestamp("2024-01-01", tz="UTC")))],
        ev[y == 2024],
        "development_2024",
    )
    dev_ic, dev_t, _ = dev[PRIMARY]
    metrics = {f"dev_{k}": {"ic": v[0], "t": v[1], "acc": v[2]} for k, v in dev.items()}
    if not np.isfinite(dev_t) or dev_t < 2.0:
        print(
            f"\nGATE 1 FAILED: 2024 {PRIMARY} t = {dev_t:+.2f} < 2. "
            "The 2025-2026 holdout stays unopened."
        )
        trial = log_external_trial(
            "earnings_release_text", "rejected_dev", HYPOTHESIS, RATIONALE, metrics, False, __file__
        )
        print(f"logged as trial #{trial.trial_id}")
        return

    cutoff = idx.searchsorted(pd.Timestamp("2025-01-01", tz="UTC"))
    hold = run(ev[ev["d0"] + 63 < cutoff], ev[y >= 2025], "holdout_2025_2026")
    h_ic, h_t, _ = hold[PRIMARY]
    metrics.update({f"holdout_{k}": {"ic": v[0], "t": v[1], "acc": v[2]} for k, v in hold.items()})
    passed = np.isfinite(h_t) and h_t >= 2.0 and np.sign(h_ic) == np.sign(dev_ic)
    status = "promoted" if passed else "rejected_holdout"
    trial = log_external_trial(
        "earnings_release_text", status, HYPOTHESIS, RATIONALE, metrics, True, __file__
    )
    print(
        f"\nGATE 2 {'PASSED' if passed else 'FAILED'}: holdout {PRIMARY} t = {h_t:+.2f} "
        f"-> trial #{trial.trial_id} {status}"
    )
    if passed:
        reg = Registry.load()
        reg.record(
            SignalRecord(
                "earnings_release_text",
                "direction",
                HYPOTHESIS,
                f"ic_{PRIMARY}",
                h_ic,
                h_t,
                "frozen 2025-2026",
                f"trial #{trial.trial_id}",
            )
        )
        reg.save()


if __name__ == "__main__":
    main()
