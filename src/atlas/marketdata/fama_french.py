"""Fama-French daily factors from Kenneth French's data library (free, CRSP-based).

Returns decimal returns (0.0123, not 1.23) on a UTC DatetimeIndex. Factors:
Mkt-RF, SMB, HML, RMW, CMA (Fama & French 2015), RF (risk-free), and Mom (momentum).
Missing values (-99.99 / -999 in the source) become NaN.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
}
_ROW = re.compile(r"^\s*(\d{8})\s*,")


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the first daily table of a French library CSV (the header row starts with ',')."""
    lines = text.replace("\r", "").split("\n")
    try:
        header_at = next(i for i, ln in enumerate(lines) if ln.startswith(","))
    except StopIteration:
        raise ValueError("no header row found in French library file") from None
    columns = [c.strip() for c in lines[header_at].split(",")[1:]]
    dates, rows = [], []
    for ln in lines[header_at + 1 :]:
        if not _ROW.match(ln):
            if dates:  # first non-data line after the table ends it
                break
            continue
        parts = [p.strip() for p in ln.split(",")]
        dates.append(parts[0])
        rows.append([float(x) for x in parts[1 : 1 + len(columns)]])
    if not rows:
        raise ValueError("no data rows found in French library file")
    values = np.array(rows, dtype=np.float64)
    values[(values <= -99.99 + 1e-9) & (values >= -99.99 - 1e-9)] = np.nan
    values[values <= -999] = np.nan
    idx = pd.to_datetime(dates, format="%Y%m%d").tz_localize("UTC").rename("timestamp")
    return pd.DataFrame(values / 100.0, index=idx, columns=columns)


def _download_csv(filename: str, client: httpx.Client) -> str:
    resp = client.get(BASE + filename, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(name).decode("latin-1")


def fetch_daily_factors(
    cache_dir: str | Path | None = "data/factors", client: httpx.Client | None = None
) -> pd.DataFrame:
    """Download (or load from cache) the 5 factors + momentum, joined on date."""
    cache = Path(cache_dir) / "ff_daily.parquet" if cache_dir else None
    own_client = client is None
    client = client or httpx.Client()
    try:
        ff5 = parse_french_csv(_download_csv(FILES["ff5_daily"], client))
        mom = parse_french_csv(_download_csv(FILES["mom_daily"], client))
    finally:
        if own_client:
            client.close()
    mom.columns = [c.strip().capitalize() for c in mom.columns]  # "Mom"
    out = ff5.join(mom, how="left")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache)
    return out


def load_cached_factors(cache_dir: str | Path = "data/factors") -> pd.DataFrame:
    return pd.read_parquet(Path(cache_dir) / "ff_daily.parquet")
