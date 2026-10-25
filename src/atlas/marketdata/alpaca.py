"""Alpaca market data and asset universe.

Notes on the free plan: the IEX feed only covers trades on the IEX exchange, so volumes
are a small fraction of the consolidated tape and intraday prices can differ slightly.
Daily closes are close to consolidated; volume-based logic (participation caps,
liquidity filters) should use the SIP feed once you are trading real size.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from atlas.config import Settings, load_settings
from atlas.marketdata.bars import Adjustment, Timeframe, normalize


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _alpaca_timeframe(tf: Timeframe) -> Any:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    return {
        Timeframe.MINUTE: TimeFrame(1, TimeFrameUnit.Minute),
        Timeframe.FIVE_MINUTE: TimeFrame(5, TimeFrameUnit.Minute),
        Timeframe.HOUR: TimeFrame(1, TimeFrameUnit.Hour),
        Timeframe.DAY: TimeFrame(1, TimeFrameUnit.Day),
    }[tf]


class AlpacaBarProvider:
    name = "alpaca"

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        batch_size: int = 100,
    ) -> None:
        self.settings = settings or load_settings()
        self.batch_size = batch_size
        if client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            key, secret = self.settings.require_alpaca()
            client = StockHistoricalDataClient(key, secret)
        self._client = client

    def get_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = Timeframe.DAY,
        adjustment: Adjustment = Adjustment.ALL,
    ) -> dict[str, pd.DataFrame]:
        from alpaca.data.enums import Adjustment as AlpacaAdjustment
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest

        out: dict[str, pd.DataFrame] = {}
        for batch in _chunks(sorted(set(symbols)), self.batch_size):
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                start=start,
                end=end,
                timeframe=_alpaca_timeframe(timeframe),
                adjustment=AlpacaAdjustment(adjustment.value),
                feed=DataFeed(self.settings.alpaca_feed),
            )
            df = self._client.get_stock_bars(request).df  # SDK handles pagination
            if df is None or df.empty:
                continue
            for sym, frame in df.groupby(level="symbol"):
                out[str(sym)] = normalize(frame.droplevel("symbol"))
        return out


@dataclass(frozen=True)
class AssetInfo:
    symbol: str
    name: str
    exchange: str
    tradable: bool
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool


class AlpacaUniverse:
    """Currently listed, tradable US equities.

    Warning: this is today's list. A backtest that picks names from it suffers from
    survivorship bias (companies that went bankrupt or were delisted are missing, so
    results look better than reality). Use it for live trading; for research, prefer a
    point-in-time universe or document the bias next to every result.
    """

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or load_settings()
        if client is None:
            from alpaca.trading.client import TradingClient

            key, secret = self.settings.require_alpaca()
            client = TradingClient(key, secret, paper=self.settings.alpaca_paper)
        self._client = client

    def assets(
        self,
        exchanges: tuple[str, ...] = ("NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"),
    ) -> list[AssetInfo]:
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        raw = self._client.get_all_assets(
            GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        )
        out = []
        for a in raw:
            exchange = getattr(a.exchange, "value", str(a.exchange))
            if a.tradable and exchange in exchanges:
                out.append(
                    AssetInfo(
                        a.symbol,
                        a.name or "",
                        exchange,
                        a.tradable,
                        bool(a.shortable),
                        bool(a.easy_to_borrow),
                        bool(a.fractionable),
                    )
                )
        return out
