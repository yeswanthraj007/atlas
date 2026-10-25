"""SEC filing text (10-K / 10-Q): download, clean, and extract the Risk Factors section.

Point-in-time: each filing carries EDGAR's acceptance timestamp. A filing accepted after
the 16:00 New York close is first usable on the next trading day; one accepted earlier
is usable from that day's close (we trade on closes, so still the next session).
Documents are stored as compressed plain text: data/sec/text/<CIK>/<accession>.txt.gz
"""

from __future__ import annotations

import gzip
import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_HIDDEN = re.compile(r"<(script|style|ix:header|head)\b.*?</\1>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</?(p|div|br|tr|li|h[1-6]|table)\b[^>]*>", re.I)
_WS = re.compile(r"[ \t\u00a0]+")


@dataclass(frozen=True)
class Filing:
    cik: int
    symbol: str
    form: str
    accession: str
    filed: str  # YYYY-MM-DD
    accepted: str  # ISO timestamp from EDGAR
    period: str
    document: str

    @property
    def url(self) -> str:
        return DOC_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.document)


def html_to_text(raw: str) -> str:
    """Strip markup but keep paragraph breaks (section detection needs line starts)."""
    s = _HIDDEN.sub(" ", raw)
    s = _BLOCK.sub("\n", s)
    s = _TAGS.sub(" ", s)
    s = html.unescape(s)
    s = _WS.sub(" ", s)
    lines = [ln.strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


_ITEM_1A = re.compile(r"^\s*item\s*1a\.?\s*[:\-\u2013\u2014]?\s*risk\s+factors", re.I | re.M)
_ITEM_END = re.compile(r"^\s*item\s*(1b|1c|2)\.?\s", re.I | re.M)


def risk_factors_section(text: str, min_chars: int = 2000) -> str:
    """Item 1A text. The table of contents also matches, so take the longest candidate
    span between an 'Item 1A. Risk Factors' heading and the next item heading."""
    best = ""
    for m in _ITEM_1A.finditer(text):
        end = _ITEM_END.search(text, m.end())
        span = text[m.end() : end.start() if end else len(text)]
        if len(span) > len(best):
            best = span
    return best.strip() if len(best) >= min_chars else ""


def parse_submissions(data: dict, symbol: str, forms: tuple[str, ...]) -> list[Filing]:
    r = data.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(r.get("form", [])):
        if form in forms and r["primaryDocument"][i]:
            out.append(
                Filing(
                    int(data["cik"]),
                    symbol,
                    form,
                    r["accessionNumber"][i],
                    r["filingDate"][i],
                    r["acceptanceDateTime"][i],
                    r["reportDate"][i],
                    r["primaryDocument"][i],
                )
            )
    return out


class FilingClient:
    def __init__(
        self,
        user_agent: str,
        root: str | Path = "data/sec",
        min_interval: float = 0.13,
        client: httpx.Client | None = None,
    ) -> None:
        if "@" not in (user_agent or ""):
            raise ValueError("SEC requires a User-Agent with a name and email (SEC_USER_AGENT)")
        self.root = Path(root)
        self.min_interval = min_interval
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=90,
            follow_redirects=True,
        )
        self._last = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        """Global rate limit shared by all threads (SEC: max 10 requests/second)."""
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _get(self, url: str) -> bytes:
        for attempt in range(5):
            self._throttle()
            resp = self._client.get(url)
            if resp.status_code in (429, 503) or resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp.content
        raise httpx.HTTPError(f"gave up on {url}")

    def filings(self, symbol: str, cik: int, forms: tuple[str, ...] = ("10-K",)) -> list[Filing]:
        data = json.loads(self._get(SUBMISSIONS_URL.format(cik=cik)))
        return parse_submissions(data, symbol, forms)

    def text_path(self, f: Filing) -> Path:
        return self.root / "text" / str(f.cik) / f"{f.accession}.txt.gz"

    def fetch_text(self, f: Filing) -> Path:
        path = self.text_path(f)
        if path.exists():
            return path
        raw = self._get(f.url).decode("utf-8", errors="ignore")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write(html_to_text(raw))
        tmp.replace(path)
        return path

    def download(
        self,
        ciks: dict[str, int],
        forms: tuple[str, ...] = ("10-K",),
        since: str = "2014-01-01",
        workers: int = 6,
        progress=print,
    ) -> pd.DataFrame:
        filings, failed = [], []
        for i, (sym, cik) in enumerate(sorted(ciks.items()), 1):
            try:
                filings += [f for f in self.filings(sym, cik, forms) if f.filed >= since]
            except httpx.HTTPError as exc:
                failed.append((sym, "submissions", str(exc)[:60]))
            if i % 100 == 0:
                progress(f"  indexed {i}/{len(ciks)} companies, {len(filings)} filings")
        rows = []
        unique = {f.accession: f for f in filings}  # share classes can share one filing
        done: set[str] = set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._safe_fetch, f): f for f in unique.values()}
            for n, fut in enumerate(as_completed(futures), 1):
                f = futures[fut]
                err = fut.result()
                if err is None:
                    done.add(f.accession)
                else:
                    failed.append((f.symbol, f.accession, err))
                if n % 500 == 0:
                    progress(f"  downloaded {n}/{len(filings)}")
        rows = [f.__dict__ for f in filings if f.accession in done]
        index = pd.DataFrame(rows).sort_values(["symbol", "filed"])
        index.to_parquet(self.root / "filings_index.parquet")
        progress(f"done: {len(index)} filings, {len(failed)} failures {failed[:5]}")
        return index


def read_text(root: str | Path, cik: int, accession: str) -> str:
    with gzip.open(
        Path(root) / "text" / str(cik) / f"{accession}.txt.gz", "rt", encoding="utf-8"
    ) as fh:
        return fh.read()


def load_index(root: str | Path = "data/sec") -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "filings_index.parquet")
