"""AI scout: forward-looking theme calls, grounded in the link graph, scored later.

Why forward-only: a language model's knowledge of past outcomes contaminates any backtest
of its judgement, so scout calls are only ever scored on prices AFTER they were written.
Calls are append-only JSON files in research/scout/, committed to git.

A call names a company, the direction expected relative to the S&P 500, the horizon, the
mechanism (which link or event drives it) and a confidence from 1 to 5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCOUT_DIR = Path("research/scout")
SECTORS = [
    "Information Technology",
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Real Estate",
    "Industrials",
    "Energy",
    "Materials",
    "Utilities",
]


@dataclass
class Call:
    ticker: str
    direction: str  # "outperform" | "underperform" (vs SPY)
    horizon_days: int
    mechanism: str  # e.g. "customer link: AAPL shifting chip orders to INTC"
    link_type: str  # customer | supplier | competitor | peer | sector | event
    confidence: int  # 1..5


@dataclass
class Theme:
    name: str
    sectors: list[str]
    thesis: str
    evidence: list[str]  # headlines / filings the thesis rests on
    calls: list[Call]


@dataclass
class ScoutReport:
    report_id: str
    created_at: str
    data_through: str  # last close the author could see
    author: str
    themes: list[Theme] = field(default_factory=list)

    def validate(self) -> None:
        for t in self.themes:
            unknown = set(t.sectors) - set(SECTORS)
            if unknown:
                raise ValueError(f"theme {t.name!r}: unknown sectors {sorted(unknown)}")
            for c in t.calls:
                if c.direction not in {"outperform", "underperform"}:
                    raise ValueError(f"{c.ticker}: direction must be outperform/underperform")
                if not 1 <= c.confidence <= 5 or c.horizon_days < 1:
                    raise ValueError(f"{c.ticker}: confidence 1-5 and a positive horizon")
                if len(c.mechanism) < 15:
                    raise ValueError(f"{c.ticker}: state the mechanism")


def save_report(report: ScoutReport, root: Path = SCOUT_DIR) -> Path:
    report.validate()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{report.report_id}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; scout reports are append-only")
    path.write_text(json.dumps(asdict(report), indent=2))
    return path


def load_reports(root: Path = SCOUT_DIR) -> list[ScoutReport]:
    out = []
    for p in sorted(root.glob("*.json")):
        raw = json.loads(p.read_text())
        themes = [Theme(**{**t, "calls": [Call(**c) for c in t["calls"]]}) for t in raw["themes"]]
        out.append(ScoutReport(**{**raw, "themes": themes}))
    return out


def score_reports(
    reports: list[ScoutReport], closes: pd.DataFrame, benchmark: pd.Series
) -> pd.DataFrame:
    """Excess return vs the benchmark from the first close AFTER data_through, over each
    call's horizon. Calls whose horizon has not elapsed are marked pending."""
    idx = closes.index
    rows = []
    for rep in reports:
        start = idx.searchsorted(pd.Timestamp(rep.data_through, tz="UTC"), side="right")
        for th in rep.themes:
            for c in th.calls:
                col = f"EQ:{c.ticker}"
                end = start + c.horizon_days
                row = {
                    "report": rep.report_id,
                    "theme": th.name,
                    "ticker": c.ticker,
                    "direction": c.direction,
                    "confidence": c.confidence,
                    "horizon_days": c.horizon_days,
                    "link_type": c.link_type,
                }
                if col not in closes.columns or end >= len(idx) or start >= len(idx):
                    rows.append({**row, "status": "pending", "excess": np.nan, "hit": np.nan})
                    continue
                stock = closes[col].iloc[end] / closes[col].iloc[start] - 1
                bench = benchmark.reindex(idx).iloc[end] / benchmark.reindex(idx).iloc[start] - 1
                excess = float(stock - bench)
                hit = excess > 0 if c.direction == "outperform" else excess < 0
                rows.append({**row, "status": "scored", "excess": excess, "hit": bool(hit)})
    return pd.DataFrame(rows)


def company_context(
    ticker: str,
    links: pd.DataFrame,
    as_of: pd.Timestamp,
    sectors: dict[str, str],
    headlines: list[str] | None = None,
) -> str:
    """Everything the scout should look at before making a call on one company."""
    lk = links[links["accepted"] < as_of]
    lines = [f"{ticker} ({sectors.get(f'EQ:{ticker}', 'unknown sector')})"]
    for kind in ("customer", "supplier", "competitor", "partner"):
        named = lk[(lk["source"] == ticker) & (lk["kind"] == kind)].groupby("target").size()
        named_by = lk[(lk["target"] == ticker) & (lk["kind"] == kind)].groupby("source").size()
        if len(named):
            lines.append(f"  names as {kind}: " + ", ".join(named.nlargest(6).index))
        if len(named_by):
            lines.append(f"  named as {kind} by: " + ", ".join(named_by.nlargest(6).index))
    for h in (headlines or [])[:5]:
        lines.append(f"  news: {h}")
    lines.append(
        f"  context built {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from filings before {as_of.date()}"
    )
    return "\n".join(lines)
