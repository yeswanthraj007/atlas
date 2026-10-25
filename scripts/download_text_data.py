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


def main() -> None:
    job = sys.argv[1] if len(sys.argv) > 1 else ""
    load_dotenv(".env")
    symbols = Path("data/universes/sp500.txt").read_text().split()
    t0 = time.time()
    with single_instance(f"download_{job}"):
        if job == "filings":
            from atlas.marketdata.filings import FilingClient
            from atlas.marketdata.sec import symbols_to_ciks

            FilingClient(os.environ["SEC_USER_AGENT"]).download(
                symbols_to_ciks("data/universes/sp500_constituents.csv"),
                forms=("10-K", "10-Q"),
                since="2014-01-01",
                workers=6,
            )
        elif job == "earnings":
            from atlas.marketdata.earnings_releases import EarningsReleaseDownloader
            from atlas.marketdata.filings import FilingClient
            from atlas.marketdata.sec import symbols_to_ciks

            EarningsReleaseDownloader(FilingClient(os.environ["SEC_USER_AGENT"])).download(
                symbols_to_ciks("data/universes/sp500_constituents.csv"), since="2014-01-01"
            )
        elif job == "news":
            from atlas.marketdata.news import NewsDownloader

            NewsDownloader().download(symbols, datetime(2016, 1, 1, tzinfo=UTC), datetime.now(UTC))
        else:
            raise SystemExit("usage: download_text_data.py filings|earnings|news")
    print(f"{job} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
