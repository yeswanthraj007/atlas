"""Download and parse the SEC's quarterly insider-transaction data sets (2013 on).

    uv run python scripts/download_insider_data.py

Resumable: raw zips are cached in data/sec/insider/raw, parsed quarters in .../parts.
The latest quarter is always re-downloaded (it is still filling up).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from atlas.lock import single_instance
from atlas.marketdata.filings import FilingClient
from atlas.marketdata.insider import PAGE, ROOT, dataset_links, parse_quarter
from atlas.net import install_default_timeout


def main() -> None:
    load_dotenv(".env")
    install_default_timeout()
    client = FilingClient(os.environ["SEC_USER_AGENT"])
    raw_dir, part_dir = ROOT / "raw", ROOT / "parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)
    links = dataset_links(client._get(PAGE).decode("utf-8", "ignore"))
    latest = max(links)
    print(f"{len(links)} quarterly data sets, {min(links)} .. {latest}", flush=True)
    with single_instance("download_insider"):
        for q, hrefs in links.items():
            part = part_dir / f"{q}.parquet"
            if part.exists() and q != latest:
                continue
            raw = None
            for h in hrefs:
                try:
                    raw = client._get("https://www.sec.gov" + h)
                    break
                except Exception as exc:  # try the other folder for this quarter
                    print(f"  {q}: {h} failed ({exc})", flush=True)
            if raw is None:
                print(f"  {q}: no usable file", flush=True)
                continue
            (raw_dir / f"{q}_form345.zip").write_bytes(raw)
            df = parse_quarter(raw)
            df.to_parquet(part)
            n_buy = int((df["code"] == "P").sum())
            print(
                f"  {q}: {len(df):,} trades ({n_buy:,} purchases), "
                f"{df['symbol'].nunique()} symbols",
                flush=True,
            )
    print("done", flush=True)


if __name__ == "__main__":
    main()
