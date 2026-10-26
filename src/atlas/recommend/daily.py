"""Build the daily watchlist: refresh data, compute signals, apply the registry, write files.

Runs twice a day (pre-open and post-close). Deterministic and auditable: every number in
the report comes from stored data and tested models, and the JSON output records which
signals were allowed to rank and why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.factors.earnings import earnings_factor_frames
from atlas.factors.fundamental import fundamental_factor_frames
from atlas.marketdata.bars import Adjustment, Timeframe
from atlas.marketdata.sec import items_for, load_facts
from atlas.marketdata.store import BarStore
from atlas.ml.models import RidgeRanker
from atlas.ml.news_model import explode_by_symbol
from atlas.ml.volatility import build_vol_dataset, vol_feature_columns
from atlas.recommend.details import earnings_calendar, ticker_details, tone_model
from atlas.recommend.engine import Registry, Watchlist, build_watchlist
from atlas.recommend.social_context import social_context
from atlas.research import ResearchData, load_research_data

OUT = Path("data/reports/watchlist")


@dataclass
class DailyResult:
    as_of: pd.Timestamp
    session: str
    watchlist: Watchlist
    headlines: dict[str, list[str]]
    markdown_path: Path
    json_path: Path


def refresh_market_data(symbols: list[str], store: BarStore, progress=print) -> None:
    """Bring bars up to date. Adjusted histories are re-downloaded whole (a new split or
    dividend rewrites them); raw bars are appended."""
    from atlas.marketdata.alpaca import AlpacaBarProvider

    provider = AlpacaBarProvider()
    end = datetime.now(UTC)
    for adj, start in [
        (Adjustment.ALL, datetime(2016, 1, 1, tzinfo=UTC)),
        (Adjustment.SPLIT, datetime(2016, 1, 1, tzinfo=UTC)),
        (Adjustment.RAW, end - timedelta(days=30)),
    ]:
        bars = provider.get_bars(symbols, start, end, Timeframe.DAY, adj)
        for sym, df in bars.items():
            store.write(sym, df, Timeframe.DAY, adj)
        progress(f"  refreshed {len(bars)} symbols ({adj.value})")


def refresh_news(symbols: list[str], progress=print) -> None:
    from atlas.marketdata.news import NewsDownloader

    now = datetime.now(UTC)
    NewsDownloader().download(
        symbols, datetime(now.year, now.month, 1, tzinfo=UTC), now, progress=progress
    )


def vol_forecasts(rd: ResearchData, store: BarStore) -> pd.Series:
    """Fit the validated volatility model on all labels known today; forecast today."""
    closes, idx = rd.closes, rd.closes.index
    symbols = [c.split(":")[1] for c in closes.columns]
    facts = items_for(load_facts(), symbols)
    raw = pd.DataFrame(
        {f"EQ:{s}": store.read(s, adjustment=Adjustment.RAW)["close"] for s in symbols}
    ).reindex(idx)
    mcap = fundamental_factor_frames(facts, rd.split_closes, raw)["market_cap"]
    earn = earnings_factor_frames(facts, idx)
    extra = {"log_mcap": np.log(mcap), "days_since_filing": earn["days_since_filing"]}
    grid = idx[idx >= idx[0] + pd.Timedelta(days=380)][::5]
    data = build_vol_dataset(closes, grid.union(idx[-1:]), 21, extra, rd.benchmark, rd.membership)
    feats = vol_feature_columns(data)
    known = pd.to_datetime(data["label_end"]) <= idx[-1]
    train = data[known & data["target"].notna()]
    model = RidgeRanker(alpha=1.0).fit(train[feats], train["target"])
    today = data.xs(idx[-1], level="date")
    return pd.Series(np.exp(model.predict(today[feats])), index=today.index)


def todays_news(symbols: set[str], since: pd.Timestamp) -> tuple[pd.Series, dict[str, list[str]]]:
    from atlas.marketdata.news import load_news

    news = load_news(start=f"{since:%Y-%m}")
    news = news[news["created_at"] > since]
    rows = explode_by_symbol(news, symbols)
    counts = rows.groupby("symbol").size()
    heads = (
        rows.sort_values("created_at", ascending=False)
        .groupby("symbol")["headline"]
        .apply(lambda s: list(dict.fromkeys(s))[:3])
        .to_dict()
    )
    return counts, heads


def build_daily(session: str, refresh: bool = True, top: int = 10, progress=print) -> DailyResult:
    store = BarStore("data")
    symbols = Path("data/universes/sp500.txt").read_text().split()
    if refresh:
        refresh_market_data([*symbols, "SPY"], store, progress)
        refresh_news(symbols, progress)
    rd = load_research_data("price")
    as_of = rd.closes.index[-1]
    progress(f"  data as of close {as_of.date()}")

    registry = Registry.load()
    # entries are written by the frozen tests (scripts/holdout_*.py), never by hand
    vol = vol_forecasts(rd, store)
    vol.index = vol.index.str.replace("EQ:", "")

    since = as_of - pd.Timedelta(days=1) if session == "morning" else as_of
    counts, heads = todays_news(set(symbols), since.tz_convert("UTC"))
    earn = earnings_factor_frames(items_for(load_facts(), symbols), rd.closes.index)
    days_since = earn["days_since_filing"].iloc[-1]
    days_since.index = days_since.index.str.replace("EQ:", "")
    announced = {k: v[0] for k, v in earnings_calendar(as_of).items()}  # 8-K results dates
    days_since = days_since.where(~days_since.index.isin(announced), pd.Series(announced))
    ret_1m = rd.closes.iloc[-1] / rd.closes.iloc[-22] - 1
    context = {
        "news_today": counts.astype(float),
        "days_since_report": days_since,
        "return_1m": ret_1m.set_axis(ret_1m.index.str.replace("EQ:", "")),
    }
    signals: dict[str, pd.Series] = {}
    wl = build_watchlist(as_of.date(), signals, vol, context, registry, top)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{as_of:%Y-%m-%d}_{session}"
    md_path, json_path = OUT / f"{stem}.md", OUT / f"{stem}.json"
    md_path.write_text(wl.to_markdown(heads, top))
    payload = {
        "as_of": str(as_of.date()),
        "session": session,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ranking_signals": wl.used_signals,
        "context_signals": wl.context_signals,
        "picks": _records(wl.picks.head(top)),
        "risk": _records(wl.risk_alerts.head(top)),
        "headlines": {
            k: v
            for k, v in heads.items()
            if k in set(wl.risk_alerts.index[:top]) or k in set(wl.picks.index[:top])
        },
        "most_news": counts.sort_values(ascending=False).head(top).to_dict(),
        "social": social_context(list(wl.risk_alerts.index[:top]), set(symbols), as_of, progress),
        "details": ticker_details(
            rd,
            list(wl.risk_alerts.index[:top]),
            vol,
            days_since,
            as_of,
            tone_model(rd, progress),
        ),
        "registry": {
            n: {
                "validated": s.validated,
                "metric": s.holdout_metric,
                "value": s.holdout_value,
                "t": s.holdout_t,
                "notes": s.notes,
            }
            for n, s in registry.signals.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    return DailyResult(as_of, session, wl, heads, md_path, json_path)


def _records(df: pd.DataFrame) -> list[dict]:
    return [{"ticker": k, **{c: _jsonable(v) for c, v in row.items()}} for k, row in df.iterrows()]


def _jsonable(v: object) -> object:
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 6)
    if isinstance(v, np.integer):
        return int(v)
    return str(v)


def telegram_summary(res: DailyResult, dashboard_url: str | None = None, top: int = 5) -> str:
    wl = res.watchlist
    when = "Pre-open watchlist" if res.session == "morning" else "After-close recap"
    lines = [f"<b>Atlas · {when} · {res.as_of:%a %d %b}</b>", ""]
    if wl.picks.empty:
        lines.append("Picks: none. No direction signal has passed its frozen test yet.")
    else:
        lines.append("<b>Stocks to look at</b>")
        lines += [f"• {t}  score {r['score']:+.2f}" for t, r in wl.picks.head(top).iterrows()]
    if not wl.risk_alerts.empty:
        lines += ["", "<b>Biggest expected moves</b> (next month, annualised vol)"]
        for t, r in wl.risk_alerts.head(top).iterrows():
            extra = ""
            news = r.get("news_today")
            if isinstance(news, float) and np.isfinite(news) and news > 0:
                extra = f" · {int(news)} news today"
            lines.append(f"• {t}  {r['expected_vol']:.0%}{extra}")
    if dashboard_url:
        lines += ["", f'<a href="{dashboard_url}">Full dashboard</a>']
    lines += ["", "<i>Research watchlist, not advice.</i>"]
    return "\n".join(lines)
