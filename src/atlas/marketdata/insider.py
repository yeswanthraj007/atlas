"""SEC insider-transaction data (Forms 4), from the SEC's parsed quarterly data sets.

Source: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
We keep non-derivative open-market purchases (code P) and sales (code S) from original
Form 4 filings (amendments, 4/A, are ignored: a simplification). Point-in-time: FILING_DATE
is when the SEC received the filing; callers must not use a filing before the next trading day.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path("data/sec/insider")
PAGE = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
_LINK = re.compile(r'href="([^"]*?(\d{4}q\d)_form345\.zip)"')
COLUMNS = [
    "filing_date", "trans_date", "symbol", "issuer_cik", "owner_cik", "owner_name",
    "relationship", "title", "code", "shares", "price", "value", "aff10b5one",
]  # fmt: skip


def dataset_links(page_html: str, first: str = "2013q1") -> dict[str, list[str]]:
    """quarter -> candidate paths (newer quarters sit under a different folder)."""
    out: dict[str, list[str]] = {}
    for href, q in _LINK.findall(page_html):
        if q >= first:
            out.setdefault(q, []).append(href)
    return dict(sorted(out.items()))


def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    """Read the wanted columns; older quarters lack some (e.g. AFF10B5ONE before 2023)."""
    df = pd.read_csv(
        z.open(name), sep="\t", dtype=str, usecols=lambda c: c in cols, on_bad_lines="skip"
    )
    return df.reindex(columns=cols)


def parse_quarter(raw_zip: bytes) -> pd.DataFrame:
    """One quarter's Form 4 open-market purchases and sales as a flat table."""
    z = zipfile.ZipFile(io.BytesIO(raw_zip))
    sub = _read(z, "SUBMISSION.tsv", [
        "ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
        "ISSUERTRADINGSYMBOL", "AFF10B5ONE",
    ])  # fmt: skip
    sub = sub[sub["DOCUMENT_TYPE"] == "4"]
    tr = _read(z, "NONDERIV_TRANS.tsv", [
        "ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
        "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
    ])  # fmt: skip
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"])]
    own = _read(z, "REPORTINGOWNER.tsv", [
        "ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP",
        "RPTOWNER_TITLE",
    ])  # fmt: skip
    df = sub.merge(tr, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
    out = pd.DataFrame(
        {
            "filing_date": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
            "trans_date": pd.to_datetime(df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"),
            "symbol": df["ISSUERTRADINGSYMBOL"]
            .str.upper()
            .str.strip()
            .str.replace("-", ".", regex=False),
            "issuer_cik": df["ISSUERCIK"],
            "owner_cik": df["RPTOWNERCIK"],
            "owner_name": df["RPTOWNERNAME"],
            "relationship": df["RPTOWNER_RELATIONSHIP"].fillna(""),
            "title": df["RPTOWNER_TITLE"].fillna(""),
            "code": df["TRANS_CODE"],
            "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce"),
            "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
            "aff10b5one": df["AFF10B5ONE"].fillna("").isin(["1", "true", "True", "Y"]),
        }
    )
    out["value"] = out["shares"] * out["price"]
    # dispositions coded P/S are consistent (P acquires, S disposes); keep as reported
    return out.dropna(subset=["filing_date", "trans_date", "symbol"])[COLUMNS].reset_index(
        drop=True
    )


def load_transactions(root: str | Path = ROOT) -> pd.DataFrame:
    parts = sorted((Path(root) / "parts").glob("*.parquet"))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
