"""Daily watchlist ("top picks to look at"), governed by a signal registry.

Every signal has a registry entry with its frozen out-of-sample test result. Only signals
whose test passed (|t| >= 2 on the frozen holdout, sign as expected) can move a stock up
or down the ranking. Everything else is shown as context, never as a reason to rank.

This is a research watchlist, not investment advice and not an order list: orders still
go through the OMS and its risk checks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REGISTRY_PATH = Path("data/signal_registry.json")
MIN_T = 2.0


@dataclass
class SignalRecord:
    name: str
    description: str
    kind: str  # "direction" (predicts relative return) or "risk" (predicts move size)
    holdout_metric: str = ""
    holdout_value: float = float("nan")
    holdout_t: float = float("nan")
    tested_on: str = ""
    notes: str = ""

    @property
    def validated(self) -> bool:
        return bool(np.isfinite(self.holdout_t) and self.holdout_t >= MIN_T)


@dataclass
class Registry:
    signals: dict[str, SignalRecord] = field(default_factory=dict)

    def record(self, rec: SignalRecord) -> None:
        self.signals[rec.name] = rec

    def validated(self, kind: str | None = None) -> list[str]:
        return [
            n for n, s in self.signals.items() if s.validated and (kind is None or s.kind == kind)
        ]

    def save(self, path: Path = REGISTRY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({n: asdict(s) for n, s in self.signals.items()}, indent=2, default=str)
        )

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> Registry:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls({n: SignalRecord(**v) for n, v in raw.items()})


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0


@dataclass
class Watchlist:
    as_of: date
    picks: pd.DataFrame  # ranked by validated direction signals
    risk_alerts: pd.DataFrame  # biggest expected moves (validated risk signals)
    used_signals: list[str]
    context_signals: list[str]

    def to_markdown(self, headlines: dict[str, list[str]] | None = None, top: int = 10) -> str:
        lines = [f"# Watchlist for {self.as_of:%A %d %B %Y}", ""]
        lines.append("Research watchlist, not advice. Orders still pass the OMS risk checks.")
        lines.append("")
        lines.append(
            "**Ranking uses validated signals:** "
            + (
                ", ".join(self.used_signals)
                or "none yet (no direction signal has "
                "passed its frozen test; the list below is context only)"
            )
        )
        lines.append("**Shown for context only:** " + (", ".join(self.context_signals) or "-"))
        lines.append("")
        lines.append("## Stocks to look at")
        if self.picks.empty:
            lines.append("_No validated direction signal: nothing is ranked as a pick._")
        else:
            lines.append(_md_table(self.picks.head(top)))
        lines.append("")
        lines.append("## Biggest expected moves (volatility forecast)")
        lines.append(
            _md_table(self.risk_alerts.head(top)) if not self.risk_alerts.empty else "_none_"
        )
        if headlines:
            lines += ["", "## Headlines behind the list"]
            for sym in list(self.picks.index[:top]) + list(self.risk_alerts.index[:top]):
                for h in headlines.get(sym, [])[:2]:
                    lines.append(f"- **{sym}**: {h}")
        return "\n".join(lines) + "\n"


COLUMN_FORMATS = {"expected_vol": "{:.0%}", "return_1m": "{:+.1%}", "score": "{:+.2f}"}


def _md_table(df: pd.DataFrame) -> str:
    cols = ["ticker", *df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [_fmt(v, COLUMN_FORMATS.get(c)) for c, v in row.items()]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _fmt(v: object, fmt: str | None = None) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "-"
        if fmt:
            return fmt.format(v)
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.3g}"
    return str(v)


def build_watchlist(
    as_of: date,
    signals: dict[str, pd.Series],  # name -> score per ticker on as_of (higher = better)
    vol_forecast: pd.Series | None,  # annualised vol forecast per ticker
    context: dict[str, pd.Series] | None,  # displayed, never ranked
    registry: Registry,
    top: int = 10,
) -> Watchlist:
    used = [n for n in registry.validated("direction") if n in signals]
    shown = [n for n in signals if n not in used] + list((context or {}).keys())
    picks = pd.DataFrame()
    if used:
        z = pd.DataFrame({n: _z(signals[n]) for n in used})
        score = z.mean(axis=1, skipna=True).dropna().sort_values(ascending=False)
        picks = pd.DataFrame({"score": score})
        for n in signals:
            picks[n] = signals[n].reindex(picks.index)
    risk = pd.DataFrame()
    if vol_forecast is not None and "vol_forecast" in registry.validated("risk"):
        v = vol_forecast.dropna().sort_values(ascending=False)
        risk = pd.DataFrame({"expected_vol": v})
    for frame in (picks, risk):
        for name, series in (context or {}).items():
            if not frame.empty:
                frame[name] = series.reindex(frame.index)
    return Watchlist(as_of, picks.head(top * 3), risk.head(top * 3), used, shown)
