"""Public-attention context for the dashboard's ticker panels. Context only: none of these
sources has passed a frozen test (see atlas.marketdata.social for why)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.marketdata.social import (
    ROOT,
    ApeWisdom,
    Reddit,
    RedditNotConfiguredError,
    StockTwits,
    WikipediaViews,
    wikipedia_articles,
)

ARTICLES = ROOT / "wikipedia" / "articles.json"


def _articles(progress=print) -> dict[str, str]:
    if ARTICLES.exists():
        return json.loads(ARTICLES.read_text())
    progress("  mapping tickers to Wikipedia articles (Wikidata)")
    a = wikipedia_articles()
    ARTICLES.parent.mkdir(parents=True, exist_ok=True)
    ARTICLES.write_text(json.dumps(a, indent=0, sort_keys=True))
    return a


def _recent(source: str, days: int, as_of: pd.Timestamp) -> pd.DataFrame:
    files = sorted((ROOT / source).glob("*.parquet"))[-days:]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "created_at" in df:
        df = df[pd.to_datetime(df["created_at"], utc=True) > as_of - pd.Timedelta(days=days)]
    return df


def _safe(fn, progress, label):
    try:
        return fn()
    except Exception as exc:  # one failing source must never break the daily report
        progress(f"  {label}: skipped ({type(exc).__name__}: {exc})")
        return None


def social_context(
    tickers: list[str], universe: set[str], as_of: pd.Timestamp, progress=print
) -> dict[str, dict]:
    """Per ticker: Reddit buzz (ApeWisdom), StockTwits messages, Reddit posts (if
    configured), Wikipedia attention. Each source is optional and fails soft."""
    ape = _safe(lambda: ApeWisdom().fetch(), progress, "apewisdom")
    if ape is not None and not ape.empty:
        ape.to_parquet(_ensure(ROOT / "apewisdom" / f"{as_of:%Y-%m-%d}.parquet"))
        ape = ape.drop_duplicates("ticker").set_index("ticker")
    st = _safe(lambda: StockTwits(min_interval=18).snapshot(tickers), progress, "stocktwits")
    st_rows = _recent("stocktwits", 7, as_of) if st is not None else pd.DataFrame()
    reddit_ok = False
    try:
        rd = Reddit(os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET"))
        reddit_ok = _safe(lambda: rd.snapshot(universe), progress, "reddit") is not None
    except RedditNotConfiguredError:
        progress("  reddit: not configured (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET)")
    rd_rows = _recent("reddit", 7, as_of) if reddit_ok else pd.DataFrame()
    articles = _safe(lambda: _articles(progress), progress, "wikipedia mapping") or {}
    wiki = WikipediaViews()
    out: dict[str, dict] = {}
    for t in tickers:
        d: dict = {}
        if ape is not None and not ape.empty and t in ape.index:
            r = ape.loc[t]
            d["reddit_buzz"] = {
                "rank": _num(r.get("rank")),
                "mentions_24h": _num(r.get("mentions")),
                "mentions_prev_24h": _num(r.get("mentions_24h_ago")),
                "upvotes_24h": _num(r.get("upvotes")),
            }
        if not st_rows.empty:
            m = st_rows[st_rows["symbol"] == t].drop_duplicates("id")
            if len(m):
                d["stocktwits"] = {
                    "messages": len(m),
                    "bullish_tags": int((m["author_tag"] == "Bullish").sum()),
                    "bearish_tags": int((m["author_tag"] == "Bearish").sum()),
                    "chat_tone_avg": float(m["chat_tone"].mean()),
                    "examples": [
                        {"time": str(x.created_at), "text": x.body[:240], "tag": x.author_tag}
                        for x in m.sort_values("created_at", ascending=False).head(3).itertuples()
                    ],
                }
        if not rd_rows.empty:
            m = rd_rows[rd_rows["tickers"].map(lambda v, t=t: t in list(v))].drop_duplicates("id")
            if len(m):
                d["reddit_posts"] = {
                    "mentions_7d": len(m),
                    "bullish": int((m["chat_tone"] > 0).sum()),
                    "bearish": int((m["chat_tone"] < 0).sum()),
                    "examples": [
                        {"sub": x.subreddit, "text": x.text[:240], "url": x.permalink}
                        for x in m.sort_values("score", ascending=False).head(3).itertuples()
                    ],
                }
        if t in articles:
            start = (as_of - pd.Timedelta(days=95)).strftime("%Y%m%d")
            v = _safe(lambda t=t, s=start: wiki.fetch(articles[t], start=s), progress, f"wiki {t}")
            if v is not None and len(v) >= 30:
                last7, base = float(v.iloc[-7:].mean()), float(v.iloc[:-7].mean())
                d["wikipedia"] = {
                    "article": articles[t],
                    "views_7d_avg": last7,
                    "views_prior_avg": base,
                    "attention_ratio": last7 / base if base > 0 else None,
                }
        if d:
            out[t] = d
    return out


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _ensure(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
