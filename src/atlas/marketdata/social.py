"""Public-attention and social data (free sources), saved as daily snapshots.

Sources
  * ApeWisdom  - ticker mentions/upvotes across Reddit investing boards and 4chan (no key).
  * StockTwits - recent public messages per ticker; some carry the author's own
                 Bullish/Bearish tag (no key; public stream).
  * Reddit     - posts and comments from investing subreddits via the official API
                 (free for non-commercial use; needs REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET).
  * Wikipedia  - daily page views of each company's article (no key; history from 2015),
                 a researched proxy for public attention.

Nothing here is a validated signal. ApeWisdom/StockTwits/Reddit have no deep history, so
they are recorded daily from now on and shown as context; Wikipedia views can be tested
through the research lab because their history goes back to 2015.

Identification: requests carry the project's public URL, never a personal email.
"""

from __future__ import annotations

import base64
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

USER_AGENT = "atlas-research/0.1 (personal research; https://github.com/yeswanthraj007/atlas)"
ROOT = Path("data/social")

# ---------------------------------------------------------------------------------------
# Ticker detection in free text
# ---------------------------------------------------------------------------------------

# Uppercase words that are also tickers but usually mean something else in chat.
NOT_TICKERS = {
    "A",
    "I",
    "AI",
    "ALL",
    "AM",
    "AN",
    "ANY",
    "ARE",
    "AT",
    "BE",
    "BIG",
    "BY",
    "CAN",
    "CEO",
    "DD",
    "DO",
    "EOD",
    "EPS",
    "ETF",
    "FOR",
    "FUN",
    "GDP",
    "GO",
    "GOOD",
    "HAS",
    "HE",
    "IMO",
    "IN",
    "IPO",
    "IS",
    "IT",
    "ITM",
    "KEY",
    "LOW",
    "MAN",
    "ME",
    "MY",
    "NEW",
    "NOW",
    "OK",
    "ON",
    "ONE",
    "OR",
    "OTM",
    "OUT",
    "PM",
    "POST",
    "RH",
    "SEE",
    "SO",
    "TA",
    "THE",
    "TO",
    "UK",
    "UP",
    "US",
    "USA",
    "WAY",
    "WE",
    "YOLO",
    "CPI",
    "FED",
    "SEC",
    "ATH",
    "IV",
    "PE",
    "TV",
    "EV",
    "REAL",
    "TRUE",
    "CASH",
    "PLAY",
    "LIFE",
    "WELL",
    "BEST",
    "FAST",
    "OPEN",
    "LOVE",
    "TECH",
}
_CASHTAG = re.compile(r"\$([A-Za-z]{1,5})(?![A-Za-z])")
_UPPER = re.compile(r"(?<![A-Za-z$])([A-Z]{2,5})(?![A-Za-z])")


def find_tickers(text: str, universe: set[str]) -> set[str]:
    """Tickers mentioned in `text`: any $CASHTAG in the universe, plus bare uppercase words
    of 3+ letters that are in the universe and not common words."""
    found = {m.upper() for m in _CASHTAG.findall(text)} & universe
    for w in _UPPER.findall(text):
        if len(w) >= 3 and w in universe and w not in NOT_TICKERS:
            found.add(w)
    return found


# Trading-chat lexicon: transparent, deliberately small. Each hit counts once per text.
BULL = re.compile(
    r"\b(bull(ish)?|calls?|moon(ing)?|rocket|long(ing)?|buy(ing)?|breakout|squeeze|"
    r"undervalued|beat(s)?|upgrade[ds]?|rip(ping)?|ath)\b|🚀|📈",
    re.I,
)
BEAR = re.compile(
    r"\b(bear(ish)?|puts?|short(ing)?|sell(ing)?|dump(ing)?|crash(ing)?|overvalued|"
    r"miss(ed|es)?|downgrade[ds]?|bag ?holder|rug ?pull|tank(ing|ed)?)\b|📉|🌈🐻",
    re.I,
)


def chat_tone(text: str) -> int:
    """+1 bullish words only, -1 bearish only, 0 both or neither."""
    b, s = bool(BULL.search(text)), bool(BEAR.search(text))
    return int(b) - int(s)


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _save(df: pd.DataFrame, source: str, day: str | None = None, key: str | None = None) -> Path:
    """Append-only daily snapshot: data/social/<source>/<day>.parquet (merged by `key`)."""
    path = ROOT / source / f"{day or _today()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and key:
        df = pd.concat([pd.read_parquet(path), df]).drop_duplicates(key, keep="last")
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------------------
# ApeWisdom
# ---------------------------------------------------------------------------------------


class ApeWisdom:
    URL = "https://apewisdom.io/api/v1.0/filter/{filter}/page/{page}"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.http = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)

    def fetch(self, board: str = "all-stocks", max_pages: int = 10) -> pd.DataFrame:
        rows: list[dict] = []
        page, pages = 1, 1
        while page <= min(pages, max_pages):
            r = self.http.get(self.URL.format(filter=board, page=page))
            r.raise_for_status()
            d = r.json()
            pages = int(d.get("pages", 1))
            rows += d.get("results", [])
            page += 1
            time.sleep(0.5)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for c in ("rank", "mentions", "upvotes", "rank_24h_ago", "mentions_24h_ago"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["board"] = board
        df["fetched_at"] = datetime.now(UTC)
        return df

    def snapshot(self) -> Path:
        return _save(self.fetch(), "apewisdom", key="ticker")


# ---------------------------------------------------------------------------------------
# StockTwits (public symbol stream)
# ---------------------------------------------------------------------------------------


class StockTwits:
    URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

    def __init__(self, client: httpx.Client | None = None, min_interval: float = 20.0) -> None:
        self.http = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)
        self.min_interval = min_interval  # public stream allows ~200 requests/hour
        self._last = 0.0

    def fetch(self, symbol: str) -> pd.DataFrame:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.http.get(self.URL.format(symbol=symbol))
        self._last = time.monotonic()
        if r.status_code != 200:
            return pd.DataFrame()
        rows = []
        for m in r.json().get("messages", []):
            tag = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
            rows.append(
                {
                    "id": m["id"],
                    "symbol": symbol,
                    "created_at": pd.Timestamp(m["created_at"]),
                    "body": m.get("body", ""),
                    "author_tag": tag,  # "Bullish", "Bearish" or None
                    "chat_tone": chat_tone(m.get("body", "")),
                }
            )
        return pd.DataFrame(rows)

    def snapshot(self, symbols: list[str]) -> Path | None:
        frames = [self.fetch(s) for s in symbols]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return None
        return _save(pd.concat(frames), "stocktwits", key="id")


# ---------------------------------------------------------------------------------------
# Reddit (official API, application-only OAuth)
# ---------------------------------------------------------------------------------------

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options", "pennystocks"]


class RedditNotConfiguredError(RuntimeError):
    pass


class Reddit:
    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    API = "https://oauth.reddit.com"

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        client: httpx.Client | None = None,
        min_interval: float = 1.0,
    ) -> None:
        if not client_id or not client_secret:
            raise RedditNotConfiguredError(
                "set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env (free app at "
                "https://www.reddit.com/prefs/apps, type 'script')"
            )
        self.http = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)
        self._auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        self._token: str | None = None
        self._expires = 0.0
        self.min_interval = min_interval  # free tier: 100 requests/minute
        self._last = 0.0

    def _token_header(self) -> dict[str, str]:
        if not self._token or time.time() > self._expires - 60:
            r = self.http.post(
                self.TOKEN_URL,
                data={"grant_type": "client_credentials"},
                headers={"Authorization": f"Basic {self._auth}", "User-Agent": USER_AGENT},
            )
            r.raise_for_status()
            d = r.json()
            self._token, self._expires = d["access_token"], time.time() + float(d["expires_in"])
        return {"Authorization": f"Bearer {self._token}", "User-Agent": USER_AGENT}

    def _get(self, path: str, **params: Any) -> dict:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.http.get(f"{self.API}{path}", params=params, headers=self._token_header())
        self._last = time.monotonic()
        r.raise_for_status()
        return r.json()

    def latest(self, sub: str, kind: str = "new", limit: int = 100) -> list[dict]:
        """Newest posts (`kind`="new") or comments (`kind`="comments") of a subreddit."""
        path = f"/r/{sub}/new" if kind == "new" else f"/r/{sub}/comments"
        return [c["data"] for c in self._get(path, limit=limit).get("data", {}).get("children", [])]

    def collect(self, universe: set[str], subs: list[str] = SUBREDDITS) -> pd.DataFrame:
        rows = []
        for sub in subs:
            for kind in ("new", "comments"):
                for d in self.latest(sub, kind):
                    text = d.get("title", "") + "\n" + (d.get("selftext") or d.get("body") or "")
                    tickers = find_tickers(text, universe)
                    if not tickers:
                        continue
                    rows.append(
                        {
                            "id": d["name"],
                            "subreddit": sub,
                            "kind": "post" if kind == "new" else "comment",
                            "created_at": pd.Timestamp(d["created_utc"], unit="s", tz="UTC"),
                            "score": d.get("score", 0),
                            "text": text[:2000],
                            "tickers": sorted(tickers),
                            "chat_tone": chat_tone(text),
                            "permalink": "https://www.reddit.com" + d.get("permalink", ""),
                        }
                    )
        return pd.DataFrame(rows)

    def snapshot(self, universe: set[str]) -> Path | None:
        df = self.collect(universe)
        return _save(df, "reddit", key="id") if not df.empty else None


# ---------------------------------------------------------------------------------------
# Wikipedia page views (history from July 2015)
# ---------------------------------------------------------------------------------------

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
_TICKER_QUERY = """
SELECT ?ticker ?article WHERE {
  ?item p:P414 ?listing .
  ?listing ps:P414 ?exchange ; pq:P249 ?ticker .
  VALUES ?exchange { wd:Q13677 wd:Q82059 }
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
}"""


def wikipedia_articles(client: httpx.Client | None = None) -> dict[str, str]:
    """ticker -> English Wikipedia article title, from Wikidata's exchange listings
    (NYSE, Nasdaq). One query for all companies."""
    http = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120)
    r = http.get(WIKIDATA_SPARQL, params={"query": _TICKER_QUERY, "format": "json"})
    r.raise_for_status()
    out: dict[str, str] = {}
    for b in r.json()["results"]["bindings"]:
        t = b["ticker"]["value"].upper().replace("-", ".")
        title = b["article"]["value"].rsplit("/", 1)[-1]
        out.setdefault(t, title)
    return out


class WikipediaViews:
    URL = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
        "all-access/user/{article}/daily/{start}/{end}"
    )

    def __init__(self, client: httpx.Client | None = None, min_interval: float = 0.05) -> None:
        self.http = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60)
        self.min_interval = min_interval
        self._last = 0.0

    def fetch(self, article: str, start: str = "20150701", end: str | None = None) -> pd.Series:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        end = end or datetime.now(UTC).strftime("%Y%m%d")
        r = self.http.get(self.URL.format(article=article, start=start, end=end))
        self._last = time.monotonic()
        if r.status_code == 404:
            return pd.Series(dtype=float)
        r.raise_for_status()
        items = r.json().get("items", [])
        return pd.Series(
            {pd.Timestamp(i["timestamp"][:8], tz="UTC"): float(i["views"]) for i in items}
        )

    def history(self, tickers: list[str], articles: dict[str, str], progress=print) -> Path:
        """Full daily history per ticker with an article -> data/social/wikipedia/views.parquet."""
        cols = {}
        for n, t in enumerate(tickers, 1):
            if t in articles:
                s = self.fetch(articles[t])
                if len(s):
                    cols[t] = s
            if n % 100 == 0:
                progress(f"  wikipedia views: {n}/{len(tickers)} tickers")
        path = ROOT / "wikipedia" / "views.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cols).sort_index().to_parquet(path)
        return path
