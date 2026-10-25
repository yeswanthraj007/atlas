"""Timestamped company news from Alpaca (Benzinga and other sources), 2015 onward.

Stored one Parquet file per month in data/news/, so downloads resume where they stopped.
Each article keeps its creation timestamp (UTC): the moment it became public, which is
what makes point-in-time backtests of news possible.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.config import Settings, load_settings

COLUMNS = ["id", "created_at", "updated_at", "headline", "summary", "source", "symbols"]


def _month_starts(start: datetime, end: datetime) -> list[datetime]:
    out, cur = [], datetime(start.year, start.month, 1, tzinfo=UTC)
    while cur < end:
        out.append(cur)
        cur = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=UTC)
    return out


class NewsDownloader:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        root: str | Path = "data/news",
        min_interval: float = 1.0,
    ) -> None:
        self.settings = settings or load_settings()
        if client is None:
            from alpaca.data.historical.news import NewsClient

            key, secret = self.settings.require_alpaca()
            client = NewsClient(key, secret)
        self._client = client
        self.root = Path(root)
        self.min_interval = min_interval  # free plan: 200 requests / minute
        self._last = 0.0

    def _fetch_window(self, symbols: list[str], start: datetime, end: datetime) -> list[Any]:
        """All articles for these symbols in [start, end). The SDK follows page tokens
        itself when no `limit` is given; we retry the whole window on errors."""
        from alpaca.data.requests import NewsRequest

        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self._client.get_news(
                    NewsRequest(
                        symbols=",".join(symbols),
                        start=start,
                        end=end,
                        include_content=False,
                        exclude_contentless=False,
                    )
                )
                self._last = time.monotonic()
                return list(resp.data.get("news", []))
            except Exception:  # rate limit or transient error: back off and retry
                time.sleep(5 * 2**attempt)
        raise RuntimeError(f"news request failed repeatedly for {start:%Y-%m-%d}")

    def fetch(
        self, symbols: list[str], start: datetime, end: datetime, chunk: int = 100
    ) -> pd.DataFrame:
        rows: dict[int, dict] = {}
        for i in range(0, len(symbols), chunk):
            for n in self._fetch_window(symbols[i : i + chunk], start, end):
                rows[n.id] = {
                    "id": n.id,
                    "created_at": n.created_at,
                    "updated_at": n.updated_at,
                    "headline": n.headline or "",
                    "summary": n.summary or "",
                    "source": n.source or "",
                    "symbols": list(n.symbols or []),
                }
        df = pd.DataFrame(list(rows.values()), columns=COLUMNS)
        if df.empty:
            return df
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
        return df.sort_values("created_at")

    def download(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        progress: Callable[[str], None] = print,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        for m in _month_starts(start, end):
            path = self.root / f"{m:%Y-%m}.parquet"
            month_end = datetime(m.year + (m.month == 12), m.month % 12 + 1, 1, tzinfo=UTC)
            complete = month_end <= now
            if path.exists() and complete:
                continue
            df = self.fetch(symbols, m, min(month_end, end))
            df.to_parquet(path)
            progress(f"  {m:%Y-%m}: {len(df):,} articles")


def load_news(
    root: str | Path = "data/news", start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    files = sorted(Path(root).glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= start[:7]]
    if end:
        files = [f for f in files if f.stem <= end[:7]]
    frames = [pd.read_parquet(f) for f in files]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    return df.sort_values("created_at").reset_index(drop=True)
