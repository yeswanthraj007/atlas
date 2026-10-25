"""Local Parquet bar store: data/bars/<adjustment>/<timeframe>/<SYMBOL>.parquet

Writes are atomic (temp file + rename), so a crash mid-download never leaves a
half-written file that a backtest could silently read.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from atlas.marketdata.bars import Adjustment, Timeframe, normalize


class BarStore:
    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)

    def _path(self, symbol: str, timeframe: Timeframe, adjustment: Adjustment) -> Path:
        safe = symbol.replace("/", "_")
        return self.root / "bars" / adjustment.value / timeframe.value / f"{safe}.parquet"

    def write(
        self,
        symbol: str,
        bars: pd.DataFrame,
        timeframe: Timeframe = Timeframe.DAY,
        adjustment: Adjustment = Adjustment.ALL,
        merge: bool | None = None,
    ) -> Path:
        """Save bars. RAW data merges with what is stored; adjusted data replaces it,
        because a new split or dividend rewrites the whole adjusted history."""
        path = self._path(symbol, timeframe, adjustment)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = normalize(bars)
        if merge is None:
            merge = adjustment is Adjustment.RAW
        if merge and path.exists():
            new = normalize(pd.concat([pd.read_parquet(path), new]))
        tmp = path.with_suffix(".parquet.tmp")
        new.to_parquet(tmp)
        os.replace(tmp, path)
        return path

    def read(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.DAY,
        adjustment: Adjustment = Adjustment.ALL,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        path = self._path(symbol, timeframe, adjustment)
        if not path.exists():
            raise FileNotFoundError(f"no {adjustment}/{timeframe} bars stored for {symbol}")
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start).tz_convert("UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end).tz_convert("UTC")]
        return df

    def symbols(
        self, timeframe: Timeframe = Timeframe.DAY, adjustment: Adjustment = Adjustment.ALL
    ) -> list[str]:
        folder = self.root / "bars" / adjustment.value / timeframe.value
        return sorted(p.stem for p in folder.glob("*.parquet")) if folder.exists() else []

    def last_timestamp(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.DAY,
        adjustment: Adjustment = Adjustment.RAW,
    ) -> pd.Timestamp | None:
        try:
            df = self.read(symbol, timeframe, adjustment)
        except FileNotFoundError:
            return None
        return df.index[-1] if len(df) else None
