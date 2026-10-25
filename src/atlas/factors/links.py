"""Economic links between companies, extracted from 10-K text (point-in-time).

A link is a sentence in company A's 10-K that names company B together with relationship
words. Types:
  * customer   - "B accounted for 14% of our revenue", "our largest customer, B"
  * supplier   - "we rely on B as a sole source", "B manufactures our chips", "foundry"
  * competitor - "we compete with B"
  * partner    - "collaboration / agreement / partnership with B"
Each link is dated by the 10-K's EDGAR acceptance time and is usable only afterwards.
Cohen & Frazzini (2008) show news travels slowly along customer-supplier links.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from atlas.marketdata.filings import read_text

LINK_WORDS = {
    "customer": re.compile(
        r"\bcustomers?\b|accounted for (?:approximately )?\d+(?:\.\d+)?\s?%"
        r"|% of (?:our )?(?:total |net )?(?:revenues?|sales)",
        re.I,
    ),
    "supplier": re.compile(
        r"\bsuppliers?\b|sole[- ]source|single[- ]source|\bfoundr(?:y|ies)\b"
        r"|manufactures? (?:our|certain|substantially)|contract manufactur",
        re.I,
    ),
    "competitor": re.compile(r"\bcompet(?:e|es|ing|itor|itors|ition)\b", re.I),
    "partner": re.compile(
        r"\bpartner(?:s|ship)?\b|\bcollaborat|\bjoint venture\b|\balliance\b", re.I
    ),
}
_SENT = re.compile(r"(?<=[.!?])\s+")
_SUFFIX = re.compile(
    r",?\s+(?:inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|"
    r"plc|n\.v\.|s\.a\.|holdings?|group|the)$",
    re.I,
)
# One-word names that are also everyday English words: require the full legal name.
AMBIGUOUS = {
    "target",
    "ball",
    "gap",
    "visa",
    "carrier",
    "progressive",
    "southern",
    "general",
    "public",
    "mosaic",
    "dominion",
    "waters",
    "equity",
    "news",
    "global",
    "united",
    "american",
    "first",
    "align",
    "fair",
    "best",
    "booking",
    "match",
    "insulet",
    "regency",
    "republic",
    "realty",
    "welltower",
    "eaton",
    "linde",
    "tapestry",
    "copart",
    "monster",
    "masco",
    "quanta",
    "solventum",
    "veralto",
    "vistra",
    "trane",
    "jacobs",
    "loews",
    "cintas",
    "corning",
    "western",
    "digital",
    "intuit",
}


def company_aliases(constituents: pd.DataFrame) -> dict[str, list[str]]:
    """symbol -> name patterns, e.g. 'Apple Inc.' -> ['Apple Inc.', 'Apple']."""
    out: dict[str, list[str]] = {}
    for sym, name in zip(constituents["Symbol"], constituents["Security"], strict=True):
        sym = str(sym).replace("-", ".")
        full = str(name).strip()
        short = full
        for _ in range(3):
            short = _SUFFIX.sub("", short).strip()
        short = re.sub(r"\s*\(.*?\)\s*", " ", short).strip()
        names = {full}
        if short and short.lower() not in AMBIGUOUS and len(short) >= 3:
            names.add(short)
        out[sym] = sorted(names, key=len, reverse=True)
    return out


@dataclass(frozen=True)
class Link:
    source: str  # the filer
    target: str  # the company it names
    kind: str
    accepted: str
    sentence: str


def extract_links(
    text: str, source: str, accepted: str, finder: tuple[re.Pattern[str], dict[str, str]]
) -> list[Link]:
    """Scan each sentence once with a single combined name pattern."""
    pattern, name_to_symbol = finder
    links = []
    for sent in _SENT.split(text):
        if len(sent) > 1500:
            continue
        targets = {name_to_symbol[m.group(0)] for m in pattern.finditer(sent)}
        targets.discard(source)
        if not targets:
            continue
        kinds = [k for k, words in LINK_WORDS.items() if words.search(sent)]
        for target in targets:
            for kind in kinds:
                links.append(Link(source, target, kind, accepted, sent.strip()[:300]))
    return links


def name_finder(aliases: dict[str, list[str]]) -> tuple[re.Pattern[str], dict[str, str]]:
    """One case-sensitive whole-word pattern for every company name (longest first), plus
    a map from the matched name back to its ticker. Names shared by two tickers (share
    classes) map to the first."""
    name_to_symbol: dict[str, str] = {}
    for sym, names in aliases.items():
        for n in names:
            name_to_symbol.setdefault(n, sym)
    alternation = "|".join(re.escape(n) for n in sorted(name_to_symbol, key=len, reverse=True))
    return re.compile(r"(?<![\w&])(?:" + alternation + r")(?![\w&])"), name_to_symbol


def build_link_table(
    index: pd.DataFrame, constituents: pd.DataFrame, root: str = "data/sec", progress=print
) -> pd.DataFrame:
    finder = name_finder(company_aliases(constituents))
    rows: list[Link] = []
    tenk = index[index["form"] == "10-K"].drop_duplicates("accession")
    for n, r in enumerate(tenk.itertuples(), 1):
        try:
            text = read_text(root, r.cik, r.accession)
        except FileNotFoundError:
            continue
        rows += extract_links(text, r.symbol, r.accepted, finder)
        if n % 500 == 0:
            progress(f"  scanned {n}/{len(tenk)} 10-Ks, {len(rows):,} link sentences")
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df
    df["accepted"] = pd.to_datetime(df["accepted"], utc=True)
    return df


def link_matrix(
    links: pd.DataFrame, as_of: pd.Timestamp, kind: str, min_mentions: int = 2
) -> dict[str, Counter[str]]:
    """source -> Counter(target -> mentions) from each source's latest 10-K before as_of."""
    lk = links[(links["accepted"] < as_of) & (links["kind"] == kind)]
    if lk.empty:
        return {}
    latest = lk.groupby("source")["accepted"].transform("max")
    lk = lk[lk["accepted"] == latest]
    out: dict[str, Counter[str]] = {}
    for (src, tgt), n in lk.groupby(["source", "target"]).size().items():
        if n >= min_mentions:
            out.setdefault(src, Counter())[tgt] = int(n)
    return out
