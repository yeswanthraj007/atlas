"""Download SEC filing texts and/or Alpaca news. One instance per job (lock file).

uv run python scripts/download_text_data.py filings
uv run python scripts/download_text_data.py news
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from atlas.lock import single_instance

CONSTITUENTS = [
    "data/universes/sp500_constituents.csv",
    "data/universes/sp400_constituents.csv",
    "data/universes/sp600_constituents.csv",
]


def main() -> None:
    job = sys.argv[1] if len(sys.argv) > 1 else ""
    universe = sys.argv[2] if len(sys.argv) > 2 else "data/universes/sp500.txt"
    forms = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("10-K", "10-Q")
    load_dotenv(".env")
    symbols = Path(universe).read_text().split()
    from atlas.marketdata.sec import ciks_from_files

    ciks = {
        s: c
        for s, c in ciks_from_files([p for p in CONSTITUENTS if Path(p).exists()]).items()
        if s in set(symbols)
    }
    t0 = time.time()
    with single_instance(f"download_{job}"):
        if job == "filings":
            from atlas.marketdata.filings import FilingClient

            FilingClient(os.environ["SEC_USER_AGENT"]).download(
                ciks, forms=forms, since="2014-01-01", workers=6
            )
        elif job == "earnings":
            from atlas.marketdata.earnings_releases import EarningsReleaseDownloader
            from atlas.marketdata.filings import FilingClient

            EarningsReleaseDownloader(FilingClient(os.environ["SEC_USER_AGENT"])).download(
                ciks, since="2014-01-01"
            )
        elif job == "news":
            from atlas.marketdata.news import NewsDownloader

            NewsDownloader().download(symbols, datetime(2016, 1, 1, tzinfo=UTC), datetime.now(UTC))
        else:
            raise SystemExit("usage: download_text_data.py filings|earnings|news")
    print(f"{job} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
