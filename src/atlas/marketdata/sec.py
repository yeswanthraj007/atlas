"""Point-in-time fundamentals from SEC EDGAR XBRL "company facts" (free, official).

Look-ahead rules enforced here:
  * every value is keyed by the date it was FILED, and becomes usable the next business
    day (filings often land after the close);
  * when a figure is later restated, the FIRST-reported value is kept as of its original
    filing date; the restated value only appears from its own filing date;
  * trailing-twelve-month (TTM) flows are built only from facts already filed.

SEC fair-access policy: a User-Agent with a name and contact email on every request,
and at most 10 requests per second (we stay below that).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Canonical item -> XBRL tags in priority order (first tag present for a period wins)
FLOW_TAGS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}
INSTANT_TAGS: dict[str, list[str]] = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}
SHARES_TAG = ("dei", "EntityCommonStockSharesOutstanding")

COLUMNS = ["item", "start", "end", "val", "filed", "form", "accn"]


def extract_facts(company_facts: dict) -> pd.DataFrame:
    """Flatten the items we use into rows: item, start, end, val, filed, form, accn."""
    rows: list[tuple] = []
    gaap = company_facts.get("facts", {}).get("us-gaap", {})
    for item, tags in {**FLOW_TAGS, **INSTANT_TAGS}.items():
        for priority, tag in enumerate(tags):
            for unit_rows in gaap.get(tag, {}).get("units", {}).values():
                for f in unit_rows:
                    rows.append(
                        (
                            item,
                            priority,
                            f.get("start"),
                            f["end"],
                            f["val"],
                            f["filed"],
                            f.get("form", ""),
                            f.get("accn", ""),
                        )
                    )
    ns, tag = SHARES_TAG
    for unit_rows in (
        company_facts.get("facts", {}).get(ns, {}).get(tag, {}).get("units", {}).values()
    ):
        for f in unit_rows:
            rows.append(
                (
                    "shares",
                    0,
                    None,
                    f["end"],
                    f["val"],
                    f["filed"],
                    f.get("form", ""),
                    f.get("accn", ""),
                )
            )
    df = pd.DataFrame(
        rows, columns=["item", "priority", "start", "end", "val", "filed", "form", "accn"]
    )
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    df["val"] = df["val"].astype(float)
    # Dual-class companies report one share count per class in the same filing: sum them.
    shares = (
        df[df["item"] == "shares"]
        .groupby(["end", "filed", "accn", "form"], as_index=False)["val"]
        .sum()
    )
    shares["item"], shares["priority"], shares["start"] = "shares", 0, pd.NaT
    df = pd.concat([df[df["item"] != "shares"], shares], ignore_index=True)
    # For each item & period keep the best-priority tag, then the FIRST filing of it.
    df = df.sort_values(["item", "start", "end", "priority", "filed"])
    df = df.drop_duplicates(["item", "start", "end"], keep="first")
    return df[COLUMNS].reset_index(drop=True)


def _duration(df: pd.DataFrame) -> pd.Series:
    return (df["end"] - df["start"]).dt.days


def ttm_series(facts: pd.DataFrame, item: str) -> pd.DataFrame:
    """Trailing-12-month values of a flow item: columns end, val, known (filed date).

    TTM(E) = annual value ending E, or
           = last annual (ending A) + YTD ending E - prior-year YTD ending ~E-1y,
    where the YTD period starts the day after A. Known when the last input was filed.
    """
    f = facts[(facts["item"] == item) & facts["start"].notna()].copy()
    if f.empty:
        return pd.DataFrame(columns=["end", "val", "known"])
    f["days"] = _duration(f)
    annual = f[f["days"].between(350, 380)]
    ytd = f[f["days"].between(80, 290)]
    out = [(r.end, r.val, r.filed) for r in annual.itertuples()]
    ann_by_end = {r.end: r for r in annual.itertuples()}
    for r in ytd.itertuples():
        prior_fy_end = r.start - pd.Timedelta(days=1)
        a = _near(ann_by_end, prior_fy_end, 7)
        if a is None:
            continue
        prev = ytd[
            (ytd["end"] - (r.end - pd.DateOffset(years=1))).abs().dt.days.le(7)
            & ((ytd["days"] - r.days).abs() <= 10)
        ]
        if prev.empty:
            continue
        p = prev.iloc[0]
        out.append((r.end, a.val + r.val - p["val"], max(r.filed, a.filed, p["filed"])))
    res = pd.DataFrame(out, columns=["end", "val", "known"])
    return res.sort_values(["known", "end"]).drop_duplicates("end", keep="first")


def _near(by_end: dict, target: pd.Timestamp, tol_days: int):
    for end, row in by_end.items():
        if abs((end - target).days) <= tol_days:
            return row
    return None


def instant_series(facts: pd.DataFrame, item: str) -> pd.DataFrame:
    f = facts[facts["item"] == item]
    return f.rename(columns={"filed": "known"})[["end", "val", "known"]].sort_values("known")


def point_in_time(series: pd.DataFrame, dates: pd.DatetimeIndex, lag_days: int = 1) -> pd.Series:
    """Daily series: on each date, the value of the latest-ENDING period already known.

    Known = filing date + `lag_days` business days. Among known values the one with the
    most recent period end wins, so a late amendment of an old period never replaces
    newer data.
    """
    if series.empty:
        return pd.Series(np.nan, index=dates)
    s = series.copy()
    s["usable"] = pd.to_datetime(s["known"]) + pd.offsets.BDay(lag_days)
    s = s.sort_values("usable")
    s["best_end"] = s["end"].cummax()
    s = s[s["end"] == s["best_end"]].drop_duplicates("usable", keep="last")
    idx = pd.DatetimeIndex(s["usable"])
    if dates.tz is not None:
        idx = idx.tz_localize(dates.tz)
    vals = pd.Series(s["val"].to_numpy(), index=idx)
    return vals.reindex(vals.index.union(dates)).ffill().reindex(dates)


class SecClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path = "data/sec",
        min_interval: float = 0.15,
        client: httpx.Client | None = None,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with a name and contact email "
                "(set SEC_USER_AGENT in .env)"
            )
        self.cache_dir = Path(cache_dir)
        self.min_interval = min_interval
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=60
        )
        self._last = 0.0

    def company_facts(self, cik: int) -> dict:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            resp = self._client.get(FACTS_URL.format(cik=cik))
            self._last = time.monotonic()
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return json.loads(resp.content)
        resp.raise_for_status()
        return {}

    def download(self, ciks: dict[str, int], progress=print) -> pd.DataFrame:
        """Fetch and extract facts for {symbol: cik}; cached as one Parquet file."""
        frames, failed = [], []
        for i, (sym, cik) in enumerate(sorted(ciks.items()), 1):
            try:
                df = extract_facts(self.company_facts(cik))
                df.insert(0, "symbol", sym)
                frames.append(df)
            except httpx.HTTPError as exc:
                failed.append((sym, str(exc)[:80]))
            if i % 50 == 0:
                progress(f"  {i}/{len(ciks)}")
        out = pd.concat(frames, ignore_index=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(self.cache_dir / "facts.parquet")
        if failed:
            progress(f"  failed: {failed[:10]}")
        return out


def load_facts(cache_dir: str | Path = "data/sec") -> pd.DataFrame:
    return pd.read_parquet(Path(cache_dir) / "facts.parquet")


def symbols_to_ciks(constituents_csv: str) -> dict[str, int]:
    df = pd.read_csv(constituents_csv)
    sym = df["Symbol"].astype(str).str.replace("-", ".", regex=False)
    return dict(zip(sym, df["CIK"].astype(int), strict=True))


def items_for(facts: pd.DataFrame, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    return {s: g.drop(columns="symbol") for s, g in facts.groupby("symbol") if s in set(symbols)}
