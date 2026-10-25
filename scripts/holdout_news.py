"""Frozen news-sentiment test: learn from 2016-2024 news, predict 2025-2026 per ticker.

    uv run python scripts/holdout_news.py --tickers AAPL MSFT NVDA AMZN TSLA COST JPM XOM

All returns are market-adjusted (minus SPY over the same window). The tradeable test buys
at the NEXT open after the news day's close; the same-day move is shown only as a
reference (it already happened by the time we can act).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atlas.marketdata.news import load_news
from atlas.marketdata.store import BarStore
from atlas.ml.news_model import (
    NewsSentimentModel,
    daily_scores,
    explode_by_symbol,
    identity_tokens,
    news_day_index,
)
from atlas.research import load_research_data

OUT = Path("data/reports/holdout_news")
COST_ROUND_TRIP = 0.0012  # 5 bps slippage + 1 bp fees, each way


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument(
        "--tickers",
        nargs="*",
        default=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "COST", "JPM", "XOM"],
    )
    args = ap.parse_args()
    args.tickers = [t.upper() for x in args.tickers for t in re.split(r"[\s,]+", x) if t]
    OUT.mkdir(parents=True, exist_ok=True)

    rd = load_research_data("price")
    idx = rd.closes.index
    syms = [c.split(":")[1] for c in rd.closes.columns]
    close = rd.closes.set_axis(syms, axis=1)
    opens = pd.DataFrame({k.split(":")[1]: v["open"] for k, v in rd.bars.items()}).reindex(idx)
    spy = BarStore("data").read("SPY").reindex(idx)
    c, o = close.to_numpy(), opens.reindex(columns=syms).to_numpy()
    sc, so = spy["close"].to_numpy(), spy["open"].to_numpy()
    col = {s: i for i, s in enumerate(syms)}

    news = load_news()
    rows = explode_by_symbol(news, set(syms))
    rows["d0"] = news_day_index(rows["created_at"], idx)
    rows = rows[(rows["d0"] >= 1) & (rows["d0"] + 5 < len(idx))].copy()
    j = rows["symbol"].map(col).to_numpy()
    d0 = rows["d0"].to_numpy()

    def abn(a: np.ndarray, b: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
        return a / b - sa / sb

    rows["reaction"] = abn(c[d0 + 1, j], c[d0 - 1, j], sc[d0 + 1], sc[d0 - 1])  # training label
    train_end = idx.searchsorted(pd.Timestamp(args.train_end, tz="UTC"), side="right") - 1
    train = rows[(rows["d0"] + 1 <= train_end) & rows["reaction"].notna()]
    test = rows[rows["d0"] > train_end]
    print(
        f"news: {len(news):,} articles | {len(train):,} training article-stock pairs "
        f"(to {idx[train_end].date()}) | {len(test):,} test pairs "
        f"({idx[test['d0'].min()].date()} .. {idx[test['d0'].max()].date()}) | model FROZEN"
    )

    masked = identity_tokens(train)
    model = NewsSentimentModel(stop_words=masked).fit(train["text"], train["reaction"])
    pos_w, neg_w = model.top_words(15)
    print(f"masked {len(masked):,} company-identity words")
    print("learned positive words:", ", ".join(pos_w))
    print("learned negative words:", ", ".join(neg_w))

    day = daily_scores(test, model.score(test["text"]))
    dd, jj = day["d0"].to_numpy(), day["symbol"].map(col).to_numpy()
    day["same_day"] = abn(c[dd, jj], c[dd - 1, jj], sc[dd], sc[dd - 1])
    day["next_day"] = abn(c[dd + 1, jj], o[dd + 1, jj], sc[dd + 1], so[dd + 1])
    day["next_5d"] = abn(c[dd + 5, jj], o[dd + 1, jj], sc[dd + 5], so[dd + 1])
    day["date"] = idx[dd].date
    day = day.dropna(subset=["next_day"])
    strong = day["score"].abs() >= day["score"].abs().quantile(0.7)

    print(
        f"\n=== 1. ACCURACY on 2025-2026 ({len(day):,} stock-days with news, "
        f"{day['date'].nunique()} days) ==="
    )
    print("  'right' = sign of sentiment matches sign of the market-adjusted return\n")
    for name, colname, note in [
        ("Same day (reference, NOT tradeable)", "same_day", "the news already moved it"),
        ("Next day, open to close (tradeable)", "next_day", "buy at next open"),
        ("Next 5 days from next open (tradeable)", "next_5d", ""),
    ]:
        right = np.sign(day["score"]) == np.sign(day[colname])
        r_strong = right[strong]
        print(
            f"  {name:42s} all news {right.mean():.1%} | strongest 30% {r_strong.mean():.1%}"
            f"   {note}"
        )

    def daily_ls(colname: str) -> pd.Series:
        def one(g: pd.DataFrame) -> float:
            if len(g) < 10:
                return np.nan
            q = g["score"].rank(pct=True)
            return g.loc[q >= 0.8, colname].mean() - g.loc[q <= 0.2, colname].mean()

        return day.groupby("date").apply(one).dropna()

    def daily_ic(colname: str) -> pd.Series:
        return (
            day.groupby("date")
            .apply(
                lambda g: (
                    stats.spearmanr(g["score"], g[colname]).statistic if len(g) >= 10 else np.nan
                )
            )
            .dropna()
        )

    print("\n=== 2. Ranking skill and a daily long/short test ===")
    for colname, label in [
        ("same_day", "same day (reference)"),
        ("next_day", "next day"),
        ("next_5d", "next 5 days"),
    ]:
        ic, ls = daily_ic(colname), daily_ls(colname)
        net = ls - COST_ROUND_TRIP * 2  # long leg and short leg each pay a round trip
        print(
            f"  {label:22s} IC {ic.mean():+.3f} (t {t_stat(ic):+.2f}) | top-20% minus "
            f"bottom-20%: {ls.mean():+.2%} per trade (t {t_stat(ls):+.2f}), "
            f"after costs {net.mean():+.2%}"
        )

    print("\n=== 3. Per ticker (next-day, tradeable) ===")
    recs = []
    for t in args.tickers:
        d = day[day["symbol"] == t]
        if d.empty:
            continue
        right = np.sign(d["score"]) == np.sign(d["next_day"])
        posn, negn = d[d["score"] > 0], d[d["score"] < 0]
        recs.append(
            {
                "ticker": t,
                "news_days": len(d),
                "direction_right": right.mean(),
                "after_positive_news": posn["next_day"].mean() if len(posn) else np.nan,
                "after_negative_news": negn["next_day"].mean() if len(negn) else np.nan,
                "same_day_right": (np.sign(d["score"]) == np.sign(d["same_day"])).mean(),
            }
        )
    per = pd.DataFrame(recs).set_index("ticker")
    print(
        per.to_string(
            formatters={
                "direction_right": "{:.0%}".format,
                "after_positive_news": "{:+.2%}".format,
                "after_negative_news": "{:+.2%}".format,
                "same_day_right": "{:.0%}".format,
            }
        )
    )
    day.to_csv(OUT / "daily_scores.csv", index=False)
    per.to_csv(OUT / "per_ticker.csv")
    print(f"\nCSV written to {OUT}/")


if __name__ == "__main__":
    main()
