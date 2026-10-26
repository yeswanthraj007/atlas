"""Per-ticker detail for the dashboard: what the news says, why a big move is expected, how
big, and what the price has already done.

Everything here is description, not a direction forecast. The expected move comes from
the validated volatility model; "why" lists the measured inputs behind it; news tone comes
from a model trained on how prices reacted to past articles (a description of the news,
which failed its test as a predictor of next-day returns).
"""

from __future__ import annotations

import pickle
import time
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.ml.news_model import (
    NewsSentimentModel,
    explode_by_symbol,
    identity_tokens,
    news_day_index,
)
from atlas.ml.volatility import log_realized_vol
from atlas.research import ResearchData

TONE_MODEL = Path("data/models/news_tone.pkl")
TONE_MAX_AGE_DAYS = 7
REPORT_CYCLE_DAYS = 91  # US companies report quarterly


def tone_model(rd: ResearchData, progress=print) -> NewsSentimentModel:
    """News-tone model: words that came with up vs down market-adjusted reactions
    (close before the news to the close after it). Retrained weekly on all past news."""
    if TONE_MODEL.exists() and time.time() - TONE_MODEL.stat().st_mtime < TONE_MAX_AGE_DAYS * 86400:
        return pickle.loads(TONE_MODEL.read_bytes())
    from atlas.marketdata.news import load_news

    idx = rd.closes.index
    syms = [c.split(":")[1] for c in rd.closes.columns]
    c = rd.closes.set_axis(syms, axis=1).to_numpy()
    spy = rd.benchmark.reindex(idx).to_numpy()
    rows = explode_by_symbol(load_news(), set(syms))
    rows["d0"] = news_day_index(rows["created_at"], idx)
    rows = rows[(rows["d0"] >= 1) & (rows["d0"] + 1 < len(idx))]
    j = rows["symbol"].map({s: i for i, s in enumerate(syms)}).to_numpy()
    d0 = rows["d0"].to_numpy()
    rows["reaction"] = c[d0 + 1, j] / c[d0 - 1, j] - spy[d0 + 1] / spy[d0 - 1]
    rows = rows[rows["reaction"].notna()]
    progress(f"  training news-tone model on {len(rows):,} article-stock pairs")
    model = NewsSentimentModel(stop_words=identity_tokens(rows)).fit(rows["text"], rows["reaction"])
    TONE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    TONE_MODEL.write_bytes(pickle.dumps(model))
    return model


def _p_abs_above(threshold: float, sigma: float) -> float:
    """P(|move| > threshold) for a normal move with standard deviation sigma."""
    if sigma <= 0:
        return 0.0
    return float(1.0 - erf(threshold / (sigma * sqrt(2.0))))


def earnings_calendar(as_of: pd.Timestamp, path: str | Path = "data/sec/earnings_index.parquet"):
    """symbol -> (days since the last earnings release, estimated days to the next one).
    Uses 8-K Item 2.02 announcement times (not the later 10-Q filing), and each company's
    own median gap between its last few announcements (clipped to 60-120 days)."""
    p = Path(path)
    if not p.exists():
        return {}
    e = pd.read_parquet(p, columns=["symbol", "accepted"])
    e["accepted"] = pd.to_datetime(e["accepted"], utc=True)
    e = e[e["accepted"] <= as_of].sort_values("accepted")
    out = {}
    for sym, g in e.groupby("symbol"):
        t = g["accepted"].drop_duplicates()
        gaps = t.diff().dt.days.dropna().tail(4)
        cycle = float(np.clip(gaps.median(), 60, 120)) if len(gaps) else float(REPORT_CYCLE_DAYS)
        since = float((as_of - t.iloc[-1]).days)
        out[sym] = (since, max(0.0, cycle - since))
    return out


def _ret(c: pd.Series, n: int, skip: int = 0) -> float | None:
    """Return over the last n days, ending `skip` days ago (None if history is too short)."""
    if len(c) <= n:
        return None
    return float(c.iloc[-1 - skip] / c.iloc[-1 - n] - 1)


def ticker_details(
    rd: ResearchData,
    tickers: list[str],
    expected_vol: pd.Series,
    days_since_report: pd.Series,
    as_of: pd.Timestamp,
    tone: NewsSentimentModel | None,
    lookback_days: int = 7,
    max_headlines: int = 8,
) -> dict[str, dict]:
    from atlas.marketdata.news import load_news

    closes = rd.closes
    lv = {w: np.exp(log_realized_vol(closes, w).iloc[-1]) for w in (5, 21, 63, 252)}
    mret = rd.benchmark.reindex(closes.index).pct_change()
    market_vol_21 = float(mret.iloc[-21:].std() * np.sqrt(252))
    market_vol_252 = float(mret.iloc[-252:].std() * np.sqrt(252))
    since = as_of - pd.Timedelta(days=lookback_days)
    news = load_news(start=f"{since:%Y-%m}")
    news = news[news["created_at"] > since]
    rows = explode_by_symbol(news, set(tickers))
    if tone is not None and len(rows):
        rows = rows.assign(tone=tone.score(rows["text"]))
    cal = earnings_calendar(as_of)
    out = {}
    for t in tickers:
        col = f"EQ:{t}"
        if col not in closes.columns:
            continue
        c = closes[col].dropna()
        vol = float(expected_vol.get(t, np.nan))
        month_sigma = vol * sqrt(21 / 252) if np.isfinite(vol) else np.nan
        if t in cal:
            dsr, next_report = cal[t]
        else:  # fall back to the last quarterly filing (later than the announcement)
            dsr = days_since_report.get(t, np.nan)
            dsr = float(dsr) if pd.notna(dsr) else None
            next_report = max(0.0, REPORT_CYCLE_DAYS - dsr) if dsr is not None else None
        mine = rows[rows["symbol"] == t].sort_values("created_at", ascending=False)
        mine = mine.drop_duplicates("headline")
        heads = [
            {
                "time": r.created_at.isoformat(timespec="minutes"),
                "source": r.source,
                "headline": r.headline,
                "summary": (r.summary or "")[:300],
                "tone": None if "tone" not in mine else round(float(r.tone), 3),
            }
            for r in mine.head(max_headlines).itertuples()
        ]
        tones = [h["tone"] for h in heads if h["tone"] is not None]
        vol_5, vol_21, vol_63, vol_252 = (float(lv[w].get(col, np.nan)) for w in (5, 21, 63, 252))
        reasons = []
        if np.isfinite(vol_21) and np.isfinite(vol_252) and vol_21 > 1.3 * vol_252:
            reasons.append(
                f"Recent swings are high: last-month volatility {vol_21:.0%} vs {vol_252:.0%} "
                "over the past year (volatility tends to persist)."
            )
        elif np.isfinite(vol_252):
            reasons.append(
                f"It is a habitually volatile stock: {vol_252:.0%} annualised over the past year."
            )
        if np.isfinite(vol_5) and np.isfinite(vol_21) and vol_5 > 1.3 * vol_21:
            reasons.append(f"The last week was rougher still ({vol_5:.0%} annualised).")
        if next_report is not None and next_report <= 30:
            reasons.append(
                f"Earnings are likely due in about {next_report:.0f} days (last results "
                f"{dsr:.0f} days ago); results days often bring large moves."
            )
        if len(mine) >= 3:
            reasons.append(
                f"Busy news flow: {len(mine)} articles in the last {lookback_days} days."
            )
        if market_vol_21 > 1.2 * market_vol_252:
            reasons.append(
                f"The whole market is choppier than usual ({market_vol_21:.0%} vs "
                f"{market_vol_252:.0%})."
            )

        out[t] = {
            "expected_vol": vol,
            "typical_month_move": month_sigma,
            "p_move_over_10pct": _p_abs_above(0.10, month_sigma)
            if np.isfinite(month_sigma)
            else None,
            "p_move_over_20pct": _p_abs_above(0.20, month_sigma)
            if np.isfinite(month_sigma)
            else None,
            "vol_5d": vol_5,
            "vol_1m": vol_21,
            "vol_3m": vol_63,
            "vol_1y": vol_252,
            "market_vol_1m": market_vol_21,
            "days_since_report": dsr,
            "next_report_in_days": next_report,
            "return_1m": _ret(c, 21),
            "return_3m": _ret(c, 63),
            "return_12_1": _ret(c, 252, skip=21),
            "news_count": len(mine),
            "news_tone_avg": float(np.mean(tones)) if tones else None,
            "headlines": heads,
            "reasons": reasons,
        }
    return out
