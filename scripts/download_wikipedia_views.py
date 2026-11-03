"""Download daily Wikipedia page views (2015-07 onward) for the S&P 1500 companies.

uv run python scripts/download_wikipedia_views.py
"""

from __future__ import annotations

from pathlib import Path

from atlas.lock import single_instance
from atlas.marketdata.social import WikipediaViews
from atlas.recommend.social_context import _articles


def main() -> None:
    tickers = Path("data/universes/sp1500.txt").read_text().split()
    with single_instance("download_wikipedia"):
        articles = _articles()
        have = [t for t in tickers if t in articles]
        print(f"{len(have)}/{len(tickers)} S&P 1500 tickers have a Wikipedia article", flush=True)
        path = WikipediaViews().history(have, articles, progress=lambda m: print(m, flush=True))
        print(f"wikipedia views saved to {path}", flush=True)


if __name__ == "__main__":
    main()
