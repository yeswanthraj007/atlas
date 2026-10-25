"""Earnings press releases from SEC Form 8-K, Item 2.02 ("Results of Operations").

Companies file their quarterly results press release as Exhibit 99.1 to an 8-K, usually
minutes after publishing it. This is a free substitute for paid earnings-call transcripts:
the same news, in the company's own words, with EDGAR's acceptance timestamp for
point-in-time use. Stored as compressed text: data/sec/earnings/<CIK>/<accession>.txt.gz
"""

from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from atlas.marketdata.filings import SUBMISSIONS_URL, FilingClient, html_to_text

ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc}.txt"
_DOC = re.compile(r"<DOCUMENT>\s*<TYPE>([^\n<]+)\n(.*?)</DOCUMENT>", re.S | re.I)


@dataclass(frozen=True)
class EarningsRelease:
    cik: int
    symbol: str
    accession: str
    filed: str
    accepted: str
    items: str


def list_releases(
    submissions: dict, symbol: str, since: str = "2014-01-01"
) -> list[EarningsRelease]:
    r = submissions.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(r.get("form", [])):
        items = r.get("items", [""] * len(r["form"]))[i] or ""
        if form == "8-K" and "2.02" in items.split(",") and r["filingDate"][i] >= since:
            out.append(
                EarningsRelease(
                    int(submissions["cik"]),
                    symbol,
                    r["accessionNumber"][i],
                    r["filingDate"][i],
                    r["acceptanceDateTime"][i],
                    items,
                )
            )
    return out


def extract_press_release(submission_text: str) -> str:
    """Plain text of the press release: the first EX-99.x exhibit (EX-99.1 preferred),
    falling back to the 8-K body when a company puts the release there instead."""
    docs = [(t.strip().upper(), body) for t, body in _DOC.findall(submission_text)]
    for wanted in ("EX-99.1", "EX-99"):
        for kind, body in docs:
            if kind.startswith(wanted):
                return html_to_text(body)
    for kind, body in docs:
        if kind.startswith("8-K"):
            return html_to_text(body)
    return ""


class EarningsReleaseDownloader:
    def __init__(self, client: FilingClient, root: str | Path = "data/sec") -> None:
        self.client = client
        self.root = Path(root)

    def path(self, rel: EarningsRelease) -> Path:
        return self.root / "earnings" / str(rel.cik) / f"{rel.accession}.txt.gz"

    def releases(self, symbol: str, cik: int, since: str) -> list[EarningsRelease]:
        data = json.loads(self.client._get(SUBMISSIONS_URL.format(cik=cik)))
        return list_releases(data, symbol, since)

    def fetch(self, rel: EarningsRelease) -> str | None:
        """Download and store one release. Returns an error string instead of raising."""
        path = self.path(rel)
        if path.exists():
            return None
        try:
            raw = self.client._get(
                ARCHIVE_URL.format(
                    cik=rel.cik, acc_nodash=rel.accession.replace("-", ""), acc=rel.accession
                )
            )
            text = extract_press_release(raw.decode("utf-8", errors="ignore"))
            if len(text) < 200:
                return "no press release text found"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{id(rel)}.tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                fh.write(text)
            tmp.replace(path)
            return None
        except Exception as exc:  # one bad filing must not stop the run
            return str(exc)[:80]

    def download(
        self, ciks: dict[str, int], since: str = "2014-01-01", workers: int = 6, progress=print
    ) -> pd.DataFrame:
        rels: list[EarningsRelease] = []
        failed: list[tuple[str, str, str]] = []
        for i, (sym, cik) in enumerate(sorted(ciks.items()), 1):
            try:
                rels += self.releases(sym, cik, since)
            except Exception as exc:
                failed.append((sym, "submissions", str(exc)[:60]))
            if i % 100 == 0:
                progress(f"  indexed {i}/{len(ciks)} companies, {len(rels)} releases")
        unique = {r.accession: r for r in rels}
        ok: set[str] = set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.fetch, r): r for r in unique.values()}
            for n, fut in enumerate(as_completed(futures), 1):
                rel, err = futures[fut], fut.result()
                (
                    ok.add(rel.accession)
                    if err is None
                    else failed.append((rel.symbol, rel.accession, err))
                )
                if n % 1000 == 0:
                    progress(f"  downloaded {n}/{len(unique)}")
        index = pd.DataFrame([asdict(r) for r in rels if r.accession in ok])
        index = index.sort_values(["symbol", "filed"]) if not index.empty else index
        index.to_parquet(self.root / "earnings_index.parquet")
        progress(f"done: {len(index)} releases, {len(failed)} failures {failed[:5]}")
        return index


def read_release(root: str | Path, cik: int, accession: str) -> str:
    with gzip.open(
        Path(root) / "earnings" / str(cik) / f"{accession}.txt.gz", "rt", encoding="utf-8"
    ) as fh:
        return fh.read()


def load_earnings_index(root: str | Path = "data/sec") -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "earnings_index.parquet")
