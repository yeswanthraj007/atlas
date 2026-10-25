"""Frozen news-sentiment test: learn from 2016-2024 news, predict 2025-2026 per ticker.

    uv run python scripts/holdout_news.py --tickers AAPL MSFT NVDA AMZN TSLA COST JPM XOM

All returns are market-adjusted (minus SPY over the same window). The tradeable test buys
at the NEXT open after the news day's close; the same-day move is shown only as a
reference (it already happened by the time we can act).
"""

from __future__ import annotations

import argparse
import hashlib
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
from atlas.recommend.engine import Registry, SignalRecord
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
        "--val-start",
        default="2024-01-01",
        help="neural head early-stops on this final training year",
    )
    ap.add_argument("--models", nargs="*", default=["tfidf", "embed_linear", "embed_mlp"])
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
    print(f"masked {len(masked):,} company-identity words")
    split = idx.searchsorted(pd.Timestamp(args.val_start, tz="UTC"))
    fit_rows, val_rows = train[train["d0"] + 1 < split], train[train["d0"] >= split]

    scores: dict[str, np.ndarray] = {}
    if "tfidf" in args.models:
        m = NewsSentimentModel(stop_words=masked).fit(train["text"], train["reaction"])
        pos_w, neg_w = m.top_words(12)
        print("tfidf positive words:", ", ".join(pos_w))
        print("tfidf negative words:", ", ".join(neg_w))
        scores["tfidf"] = m.score(test["text"])
    if {"embed_linear", "embed_mlp"} & set(args.models):
        from sklearn.linear_model import LogisticRegression

        from atlas.ml.text_embed import NeuralHead, embed_cached

        both = pd.concat([train, test])
        mask_id = hashlib.sha1(" ".join(sorted(masked)).encode()).hexdigest()[:10]
        emb = embed_cached(both, masked, f"data/news_emb/embeddings_{mask_id}.npz")
        e_train, e_test = emb[: len(train)], emb[len(train) :]
        r = train["reaction"].to_numpy()
        if "embed_linear" in args.models:
            lo, hi = np.quantile(r, 0.3), np.quantile(r, 0.7)
            keep = (r <= lo) | (r >= hi)
            clf = LogisticRegression(C=0.5, max_iter=3000).fit(e_train[keep], r[keep] >= hi)
            scores["embed_linear"] = clf.predict_proba(e_test)[:, 1] - 0.5
        if "embed_mlp" in args.models:
            fit_mask = (train["d0"] + 1 < split).to_numpy()
            head = NeuralHead().fit(
                e_train[fit_mask], r[fit_mask], e_train[~fit_mask], r[~fit_mask]
            )
            print(
                f"neural head: {len(head.history)} epochs, best validation loss "
                f"{min(v for _, v in head.history):.4f} (coin flip 0.6931)"
            )
            scores["embed_mlp"] = head.score(e_test)
    del fit_rows, val_rows

    def evaluate(name: str, article_scores: np.ndarray) -> tuple[pd.DataFrame, dict]:
        day = daily_scores(test, article_scores)
        dd, jj = day["d0"].to_numpy(), day["symbol"].map(col).to_numpy()
        day["same_day"] = abn(c[dd, jj], c[dd - 1, jj], sc[dd], sc[dd - 1])
        day["next_day"] = abn(c[dd + 1, jj], o[dd + 1, jj], sc[dd + 1], so[dd + 1])
        day["next_5d"] = abn(c[dd + 5, jj], o[dd + 1, jj], sc[dd + 5], so[dd + 1])
        day["date"] = idx[dd].date
        day = day.dropna(subset=["next_day"])
        strong = day["score"].abs() >= day["score"].abs().quantile(0.7)
        res: dict = {"model": name, "stock_days": len(day)}
        for colname in ("same_day", "next_day", "next_5d"):
            right = np.sign(day["score"]) == np.sign(day[colname])
            ic = (
                day.groupby("date")
                .apply(
                    lambda g, cn=colname: (
                        stats.spearmanr(g["score"], g[cn]).statistic if len(g) >= 10 else np.nan
                    )
                )
                .dropna()
            )

            def ls_one(g: pd.DataFrame, cn: str = colname) -> float:
                if len(g) < 10:
                    return np.nan
                q = g["score"].rank(pct=True)
                return g.loc[q >= 0.8, cn].mean() - g.loc[q <= 0.2, cn].mean()

            ls = day.groupby("date").apply(ls_one).dropna()
            res[f"{colname}_acc"] = right.mean()
            res[f"{colname}_acc_strong"] = right[strong].mean()
            res[f"{colname}_ic"] = ic.mean()
            res[f"{colname}_ic_t"] = t_stat(ic)
            res[f"{colname}_ls"] = ls.mean()
            res[f"{colname}_ls_t"] = t_stat(ls)
            res[f"{colname}_ls_net"] = ls.mean() - 2 * COST_ROUND_TRIP
        return day, res

    results, days = [], {}
    for name, sc_ in scores.items():
        days[name], res = evaluate(name, sc_)
        results.append(res)
    table = pd.DataFrame(results).set_index("model")
    print(f"\n=== 1. Model comparison, frozen 2025-2026 ({len(test):,} article-stock pairs) ===")
    for colname, label in [
        ("same_day", "SAME DAY (reference, not tradeable)"),
        ("next_day", "NEXT DAY from next open (tradeable)"),
        ("next_5d", "NEXT 5 DAYS from next open (tradeable)"),
    ]:
        print(f"\n  {label}")
        for name, row in table.iterrows():
            print(
                f"    {name:13s} direction right {row[colname + '_acc']:.1%} "
                f"(strongest 30%: {row[colname + '_acc_strong']:.1%}) | IC "
                f"{row[colname + '_ic']:+.3f} (t {row[colname + '_ic_t']:+.2f}) | top-bottom "
                f"{row[colname + '_ls']:+.2%} (t {row[colname + '_ls_t']:+.2f}), after costs "
                f"{row[colname + '_ls_net']:+.2%}"
            )
    table.to_csv(OUT / "model_comparison.csv")

    best = table["next_day_ic_t"].idxmax()
    day = days[best]
    print(f"\n=== 2. Per ticker with the best tradeable model ({best}) ===")
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
                "next_day_right": right.mean(),
                "after_positive_news": posn["next_day"].mean() if len(posn) else np.nan,
                "after_negative_news": negn["next_day"].mean() if len(negn) else np.nan,
                "same_day_right": (np.sign(d["score"]) == np.sign(d["same_day"])).mean(),
            }
        )
    per = pd.DataFrame(recs).set_index("ticker")
    print(
        per.to_string(
            formatters={
                "next_day_right": "{:.0%}".format,
                "after_positive_news": "{:+.2%}".format,
                "after_negative_news": "{:+.2%}".format,
                "same_day_right": "{:.0%}".format,
            }
        )
    )
    day.to_csv(OUT / f"daily_scores_{best}.csv", index=False)
    per.to_csv(OUT / "per_ticker.csv")

    reg = Registry.load()
    row = table.loc[best]
    reg.record(
        SignalRecord(
            name="news_sentiment",
            kind="direction",
            description=f"news sentiment ({best}), next-day open-to-close, market-adjusted",
            holdout_metric="next_day_ic",
            holdout_value=float(row["next_day_ic"]),
            holdout_t=float(row["next_day_ic_t"]),
            tested_on="2025-01..2026-10 frozen",
            notes=f"after-cost top-bottom {row['next_day_ls_net']:+.4f}/trade",
        )
    )
    reg.save()
    verdict = (
        "PASSED: may rank picks"
        if reg.signals["news_sentiment"].validated
        else ("NOT validated: shown as context only")
    )
    print(f"\nregistry: news_sentiment ({best}) t={row['next_day_ic_t']:+.2f} -> {verdict}")
    print(f"CSV written to {OUT}/")


if __name__ == "__main__":
    main()
