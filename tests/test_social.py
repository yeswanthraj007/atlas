import httpx
import pandas as pd
import pytest

from atlas.marketdata import social
from atlas.marketdata.social import (
    ApeWisdom,
    Reddit,
    RedditNotConfiguredError,
    StockTwits,
    chat_tone,
    find_tickers,
)

U = {"AAPL", "NVDA", "IT", "ALL", "ON", "AMD", "COST"}


def test_find_tickers_cashtags_and_bare_words():
    assert find_tickers("$aapl and $NVDA to the moon", U) == {"AAPL", "NVDA"}
    # common words that are also tickers count only as cashtags
    assert find_tickers("IT is ALL ON fire", U) == set()
    assert find_tickers("long $IT", U) == {"IT"}
    assert find_tickers("AMD earnings, COST too", U) == {"AMD", "COST"}
    assert find_tickers("$ZZZZ not in universe", U) == set()


def test_chat_tone_lexicon():
    assert chat_tone("buying calls, bullish 🚀") == 1
    assert chat_tone("loading puts, this will crash") == -1
    assert chat_tone("bought calls and puts") == 0
    assert chat_tone("earnings tomorrow") == 0


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_apewisdom_pages_and_types(monkeypatch):
    monkeypatch.setattr(social.time, "sleep", lambda _: None)

    def handler(req):
        page = int(str(req.url).rsplit("/", 1)[1])
        res = [
            {
                "rank": str(page),
                "ticker": f"T{page}",
                "mentions": "5",
                "upvotes": "9",
                "rank_24h_ago": "3",
                "mentions_24h_ago": "1",
            }
        ]
        return httpx.Response(
            200, json={"count": 2, "pages": 2, "current_page": page, "results": res}
        )

    df = ApeWisdom(_client(handler)).fetch()
    assert list(df["ticker"]) == ["T1", "T2"] and df["mentions"].dtype.kind in "if"


def test_stocktwits_parses_author_tags(monkeypatch):
    def handler(req):
        return httpx.Response(
            200,
            json={
                "messages": [
                    {
                        "id": 1,
                        "created_at": "2026-10-26T20:00:00Z",
                        "body": "$AAPL calls",
                        "entities": {"sentiment": {"basic": "Bullish"}},
                    },
                    {"id": 2, "created_at": "2026-10-26T20:01:00Z", "body": "meh", "entities": {}},
                ]
            },
        )

    df = StockTwits(_client(handler), min_interval=0).fetch("AAPL")
    assert df["author_tag"].iloc[0] == "Bullish" and pd.isna(df["author_tag"].iloc[1])
    assert list(df["chat_tone"]) == [1, 0]


def test_reddit_requires_credentials_and_collects_mentions():
    with pytest.raises(RedditNotConfiguredError):
        Reddit(None, None)

    def handler(req):
        if "access_token" in str(req.url):
            assert req.headers["Authorization"].startswith("Basic ")
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert req.headers["Authorization"] == "Bearer tok"
        child = {
            "name": "t3_x" + req.url.path.replace("/", "_"),
            "title": "$NVDA puts",
            "selftext": "",
            "created_utc": 1_790_000_000,
            "score": 5,
            "permalink": "/r/x",
        }
        return httpx.Response(200, json={"data": {"children": [{"data": child}]}})

    df = Reddit("id", "secret", _client(handler), min_interval=0).collect(U, subs=["stocks"])
    assert len(df) == 2 and set(df["kind"]) == {"post", "comment"}
    assert all(t == ["NVDA"] for t in df["tickers"]) and (df["chat_tone"] == -1).all()
    assert isinstance(df["created_at"].iloc[0], pd.Timestamp)
