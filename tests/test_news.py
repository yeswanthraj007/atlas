from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from atlas.marketdata.news import NewsDownloader, load_news
from atlas.ml.news_model import (
    NewsSentimentModel,
    explode_by_symbol,
    identity_tokens,
    news_day_index,
)


class FakeNews:
    def __init__(self) -> None:
        self.calls = 0

    def get_news(self, req):
        self.calls += 1
        art = SimpleNamespace(
            id=hash(req.symbols) % 1000,
            created_at=req.start,
            updated_at=req.start,
            headline="h",
            summary="s",
            source="x",
            symbols=req.symbols.split(","),
        )
        shared = SimpleNamespace(
            id=1,
            created_at=req.start,
            updated_at=req.start,
            headline="both",
            summary="",
            source="x",
            symbols=["A", "Z"],
        )
        return SimpleNamespace(data={"news": [art, shared]})


def test_downloader_dedupes_and_resumes(tmp_path) -> None:
    fake = FakeNews()
    dl = NewsDownloader(settings=SimpleNamespace(), client=fake, root=tmp_path, min_interval=0)
    dl.download(
        ["A", "B", "Z"],
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 3, 1, tzinfo=UTC),
        progress=lambda _: None,
    )
    df = load_news(tmp_path)
    assert (df["id"] == 1).sum() == 1  # article tagged with two chunks stored once
    calls = fake.calls
    dl.download(
        ["A", "B", "Z"],
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 3, 1, tzinfo=UTC),
        progress=lambda _: None,
    )
    assert fake.calls == calls  # completed months are not downloaded again


def test_news_day_respects_the_close() -> None:
    days = pd.bdate_range("2025-03-03", periods=5, tz="UTC")  # Mon..Fri
    ts = pd.Series(
        pd.to_datetime(
            [
                "2025-03-03 14:00",
                "2025-03-03 20:59",
                "2025-03-03 21:01",
                "2025-03-08 12:00",
            ],
            utc=True,
        )
    )  # 16:00 New York = 21:00 UTC in March before DST
    pos = news_day_index(ts, days)
    assert list(pos[:3]) == [0, 0, 1]  # after the close -> next trading day
    assert pos[3] == -1  # weekend after the last day in the data


def test_model_learns_words_not_company_names() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3000):
        sym = ["AAA", "BBB", "CCC"][i % 3]
        good = rng.random() < 0.5
        word = "beats raises guidance" if good else "misses cuts guidance"
        rows.append(
            {
                "symbol": sym,
                "text": f"{sym.lower()}corp {word} quarter",
                "reaction": (0.03 if good else -0.03) + rng.normal(0, 0.01),
            }
        )
    df = pd.DataFrame(rows)
    masked = identity_tokens(df, min_count=30)
    assert {"aaacorp", "bbbcorp", "ccccorp"} <= masked and "beats" not in masked
    m = NewsSentimentModel(min_df=5, stop_words=masked).fit(df["text"], df["reaction"])
    s = m.score(pd.Series(["zzzcorp beats raises guidance", "zzzcorp misses cuts guidance"]))
    assert s[0] > 0.2 and s[1] < -0.2
    pos, neg = m.top_words(3)
    assert not any("corp" in w for w in pos + neg)


def test_explode_drops_market_wraps() -> None:
    news = pd.DataFrame(
        {"headline": ["one", "wrap"], "summary": ["", ""], "symbols": [["A"], ["A", "B", "C", "D"]]}
    )
    out = explode_by_symbol(news, {"A", "B"})
    assert list(out["headline"]) == ["one"]
