"""Build the S&P MidCap 400 and SmallCap 600 universe files (current members + changes).

    uv run python scripts/build_universe.py

Writes data/universes/{sp400,sp600}_constituents.csv (Symbol, Security, GICS Sector, CIK),
{sp400,sp600}_changes.csv (dated additions/removals where the source has them),
midsmall.txt and sp1500.txt (symbol lists). CIKs come from the source table or, when
missing, from SEC's official company_tickers.json.

Limitation: constituents are TODAY's members. Companies removed since 2016 (often
because they fell) are missing, so backtests on this universe are optimistic.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

PAGES = {
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
OUT = Path("data/universes")


def main() -> None:
    load_dotenv(".env")
    headers = {"User-Agent": os.environ["SEC_USER_AGENT"]}
    sec = httpx.get(
        "https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=60
    ).json()
    cik_by_ticker = {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in sec.values()}
    lists = {}
    for name, url in PAGES.items():
        html = httpx.get(url, headers=headers, follow_redirects=True, timeout=60).text
        tables = pd.read_html(io.StringIO(html))
        cons = next(t for t in tables if "Symbol" in t.columns and "Security" in t.columns)
        cons = cons.rename(columns=str)
        cons["Symbol"] = cons["Symbol"].astype(str).str.strip().str.replace("-", ".")
        if "CIK" not in cons.columns:
            cons["CIK"] = None
        cons["CIK"] = [
            int(c) if pd.notna(c) and str(c).strip().isdigit() else cik_by_ticker.get(s)
            for s, c in zip(cons["Symbol"], cons["CIK"], strict=True)
        ]
        keep = [
            c
            for c in ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "CIK"]
            if c in cons.columns
        ]
        cons = cons[keep].dropna(subset=["CIK"])
        cons["CIK"] = cons["CIK"].astype(int)
        cons.to_csv(OUT / f"{name}_constituents.csv", index=False)
        changes = [t for t in tables if any("Added" in str(c) for c in t.columns)]
        if changes:
            ch = changes[0]
            ch.columns = [
                "_".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in ch.columns
            ]
            ch.to_csv(OUT / f"{name}_changes.csv", index=False)
        lists[name] = cons["Symbol"].tolist()
        print(f"{name}: {len(cons)} members with CIKs, changes table: {bool(changes)}")
    sp500 = (OUT / "sp500.txt").read_text().split()
    midsmall = sorted((set(lists["sp400"]) | set(lists["sp600"])) - set(sp500))
    (OUT / "midsmall.txt").write_text("\n".join(midsmall) + "\n")
    (OUT / "sp1500.txt").write_text("\n".join(sorted(set(sp500) | set(midsmall))) + "\n")
    print(
        f"mid+small (excluding S&P 500): {len(midsmall)} | S&P 1500: "
        f"{len(set(sp500) | set(midsmall))}"
    )


if __name__ == "__main__":
    main()
