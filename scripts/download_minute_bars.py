"""Download 1-minute bars (SIP, adjusted) per symbol and year to data/minute/<SYM>/<year>.parquet.

uv run python scripts/download_minute_bars.py SPY QQQ IWM --from 2016
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from atlas.lock import single_instance
from atlas.marketdata.alpaca import AlpacaBarProvider
from atlas.marketdata.bars import Adjustment, Timeframe
from atlas.net import install_default_timeout
from atlas.recommend.daily import with_retries

ROOT = Path("data/minute")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--from", dest="first", type=int, default=2016)
    args = ap.parse_args()
    load_dotenv(".env")
    install_default_timeout()
    provider = AlpacaBarProvider()
    now = datetime.now(UTC)
    with single_instance("download_minute"):
        for year in range(args.first, now.year + 1):
            start = datetime(year, 1, 1, tzinfo=UTC)
            end = min(datetime(year + 1, 1, 1, tzinfo=UTC), now)
            for sym in args.symbols:
                path = ROOT / sym / f"{year}.parquet"
                if path.exists() and year < now.year:
                    continue
                bars = with_retries(
                    lambda s=sym, a=start, b=end: provider.get_bars(
                        [s], a, b, Timeframe.MINUTE, Adjustment.ALL
                    )
                )
                df = bars.get(sym)
                if df is None or df.empty:
                    print(f"  {sym} {year}: no data", flush=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path)
                print(f"  {sym} {year}: {len(df):,} bars", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
