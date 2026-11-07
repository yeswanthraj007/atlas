"""Record today's option-implied volatility summaries for the S&P 1500.

    uv run python scripts/record_option_iv.py                 # whole S&P 1500 (about 15-25 min)
    uv run python scripts/record_option_iv.py --symbols AAPL MSFT NVDA

Run it once a day after the US close. It is resumable: re-running the same day skips
the underlyings that are already recorded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from atlas.lock import single_instance
from atlas.marketdata.options_iv import OptionIVRecorder, load_iv
from atlas.net import install_default_timeout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--universe", default="data/universes/sp1500.txt")
    args = ap.parse_args()
    load_dotenv(".env")
    install_default_timeout()
    symbols = args.symbols or Path(args.universe).read_text().split()
    aliases = {s.replace(".", ""): s for s in symbols if "." in s}  # option root BRKB -> BRK.B
    symbols = [s.replace(".", "") for s in symbols]
    with single_instance("record_option_iv"):
        path, failed = OptionIVRecorder.from_settings().record(symbols, aliases=aliases)
    if path is None:
        raise SystemExit("no prices returned; nothing recorded")
    iv = load_iv(path.parent)
    today = iv[iv["asof"] == iv["asof"].max()]
    print(
        f"saved {path}: {today['symbol'].nunique()} underlyings with options, "
        f"{len(today)} expiries, {len(failed)} failed"
    )
    if failed:
        print("failed:", ", ".join(failed[:20]))


if __name__ == "__main__":
    main()
