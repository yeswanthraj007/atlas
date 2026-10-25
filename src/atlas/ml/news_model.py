"""News sentiment learned from returns (in the spirit of Ke, Kelly & Xiu 2019).

Training: each (article, stock) pair is labelled by the stock's market-adjusted return
around the news (the day it became known and the next day). The model learns which
words come with positive vs negative reactions. This only uses past data at training time.

Prediction: an article is first usable at the close of its "news day" d0 = the first
trading day whose 16:00 New York close is at or after the article's timestamp. Trades
happen at the NEXT open (d0 + 1), so the reaction already in the price is never counted.

Identity masking: words that mostly appear in one company's news (its name, products,
CEO) are removed, so the model must learn from what the news says, not who it is about.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

NY = "America/New_York"


def close_times(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """16:00 New York on each trading day, in UTC."""
    # +12h makes the calendar date right whether days are stamped at UTC midnight or at
    # New York midnight (04:00/05:00 UTC), as vendor data often is.
    shifted = trading_days + pd.Timedelta(hours=12)
    local = pd.DatetimeIndex(shifted.tz_convert(NY).normalize().tz_localize(None))
    return (local + pd.Timedelta(hours=16)).tz_localize(NY).tz_convert("UTC")


def news_day_index(created_at: pd.Series, trading_days: pd.DatetimeIndex) -> np.ndarray:
    """Position in `trading_days` of each article's news day d0 (-1 if after the data)."""
    closes = close_times(trading_days)
    pos = closes.searchsorted(pd.DatetimeIndex(created_at), side="left")
    return np.where(pos < len(trading_days), pos, -1)


def explode_by_symbol(news: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """One row per (article, stock in our universe). Articles tagged with many stocks
    (market wraps) say little about each one, so those with > 3 symbols are dropped."""
    n = news[news["symbols"].map(len).between(1, 3)].copy()
    n = n.explode("symbols").rename(columns={"symbols": "symbol"})
    n = n[n["symbol"].isin(universe)]
    n["text"] = (n["headline"].fillna("") + ". " + n["summary"].fillna("")).str.lower()
    return n


def identity_tokens(rows: pd.DataFrame, min_count: int = 30, share: float = 0.5) -> set[str]:
    """Words whose usage is concentrated (> `share`) in a single company's articles."""
    by_word: dict[str, Counter[str]] = {}
    for sym, text in zip(rows["symbol"], rows["text"], strict=True):
        for w in set(text.split()):
            by_word.setdefault(w, Counter())[sym] += 1
    out = set()
    for w, c in by_word.items():
        total = sum(c.values())
        if total >= min_count and c.most_common(1)[0][1] / total > share:
            out.add(w)
    return out


@dataclass
class NewsSentimentModel:
    max_features: int = 60_000
    min_df: int = 20
    C: float = 0.5
    label_quantile: float = 0.3  # train on the clearest 30% up and 30% down reactions
    stop_words: set[str] = field(default_factory=set)
    _vec: TfidfVectorizer | None = field(default=None, repr=False)
    _clf: LogisticRegression | None = field(default=None, repr=False)

    def _clean(self, text: pd.Series) -> pd.Series:
        if not self.stop_words:
            return text
        sw = self.stop_words
        return text.map(lambda t: " ".join(w for w in t.split() if w not in sw))

    def fit(self, text: pd.Series, reaction: pd.Series) -> NewsSentimentModel:
        lo, hi = reaction.quantile(self.label_quantile), reaction.quantile(1 - self.label_quantile)
        keep = (reaction <= lo) | (reaction >= hi)
        y = (reaction[keep] >= hi).astype(int)
        self._vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-z][a-z\-']+\b",
        )
        x = self._vec.fit_transform(self._clean(text[keep]))
        self._clf = LogisticRegression(C=self.C, max_iter=2000, solver="liblinear").fit(x, y)
        return self

    def score(self, text: pd.Series) -> np.ndarray:
        """Probability-of-positive-reaction minus 0.5, in [-0.5, 0.5]."""
        assert self._vec is not None and self._clf is not None
        x = self._vec.transform(self._clean(text))
        return self._clf.predict_proba(x)[:, 1] - 0.5

    def top_words(self, n: int = 20) -> tuple[list[str], list[str]]:
        assert self._vec is not None and self._clf is not None
        names = np.array(self._vec.get_feature_names_out())
        coef = self._clf.coef_[0]
        order = np.argsort(coef)
        return list(names[order[-n:]][::-1]), list(names[order[:n]])


def daily_scores(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Average article score per (news day, symbol), with the article count."""
    df = rows.assign(score=scores)
    g = df.groupby(["d0", "symbol"])["score"]
    return pd.DataFrame({"score": g.mean(), "n_articles": g.size()}).reset_index()
