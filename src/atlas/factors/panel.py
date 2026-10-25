"""Factor panel: every factor, for every instrument, on every date, computed once.

Values on date t use only closes up to and including t (rolling windows, never centred
or shifted backwards). The trailing information coefficient on date d measures how
the factor ranked stocks on d against returns from d to d + horizon, so it is only
*usable* from d + horizon onward; `ic_history` enforces that.

Fundamental factors join through `add`, as already point-in-time daily frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from atlas.factors.price import MONTH, YEAR

TRADING_DAYS = YEAR


def _complete(closes: pd.DataFrame, rows: int) -> pd.DataFrame:
    """True where the last `rows` closes (inclusive) are all present."""
    return closes.notna().astype(np.int8).rolling(rows).sum().eq(rows)


def price_factor_frames(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Vectorised versions of `atlas.factors.price` (tests check they match exactly)."""
    c = closes.sort_index()
    ok_year = _complete(c, YEAR + 1)
    rets = c.pct_change(fill_method=None)
    vol = rets.rolling(YEAR).std(ddof=1) * np.sqrt(TRADING_DAYS)
    mom = c.shift(MONTH) / c.shift(YEAR) - 1.0
    return {
        "mom_12_1": mom.where(ok_year),
        "vol_adj_mom": (mom / vol.replace(0.0, np.nan)).where(ok_year),
        "reversal_1m": (-(c / c.shift(MONTH) - 1.0)).where(_complete(c, MONTH + 1)),
        "low_vol": (-vol).where(ok_year),
        "high_52w": (c / c.rolling(YEAR).max()).where(_complete(c, YEAR)),
    }


def forward_returns_daily(closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from each date's close to the close `horizon` rows later."""
    return closes.shift(-horizon) / closes - 1.0


@dataclass
class FactorPanel:
    closes: pd.DataFrame
    raw: dict[str, pd.DataFrame]
    horizon: int = MONTH
    membership: pd.DataFrame | None = None
    _ic: pd.DataFrame | None = field(default=None, repr=False)

    @classmethod
    def from_closes(
        cls,
        closes: pd.DataFrame,
        horizon: int = MONTH,
        membership: pd.DataFrame | None = None,
    ) -> FactorPanel:
        c = closes.sort_index()
        m = None
        if membership is not None:
            m = membership.reindex(index=c.index, columns=c.columns).fillna(False).astype(bool)
        return cls(c, price_factor_frames(c), horizon, m)

    @property
    def factors(self) -> list[str]:
        return list(self.raw)

    def add(self, name: str, frame: pd.DataFrame) -> None:
        """Add an externally computed point-in-time factor (e.g. fundamentals)."""
        self.raw[name] = frame.reindex(index=self.closes.index, columns=self.closes.columns)
        self._ic = None

    def at(self, ts: pd.Timestamp, eligible: frozenset[str] | None = None) -> dict[str, pd.Series]:
        """Raw factor values on `ts` for eligible instruments (NaNs dropped)."""
        out = {}
        for name, frame in self.raw.items():
            row = frame.loc[ts]
            if eligible is not None:
                row = row[row.index.isin(eligible)]
            out[name] = row.dropna()
        return out

    def daily_ic(self) -> pd.DataFrame:
        """Spearman IC of each factor on each date vs the next `horizon`-day return
        (cached). Row d becomes known only at d + horizon."""
        if self._ic is None:
            fwd = forward_returns_daily(self.closes, self.horizon)
            fwd_rank = fwd.rank(axis=1)
            cols = {}
            for name, frame in self.raw.items():
                f = frame
                if self.membership is not None:
                    f = f.where(self.membership)
                both = f.notna() & fwd.notna()
                fr = f.where(both).rank(axis=1)
                rr = fwd_rank.where(both).rank(axis=1)
                fr = fr.sub(fr.mean(axis=1), axis=0)
                rr = rr.sub(rr.mean(axis=1), axis=0)
                num = (fr * rr).sum(axis=1)
                den = np.sqrt((fr**2).sum(axis=1) * (rr**2).sum(axis=1))
                ic = num / den.replace(0.0, np.nan)
                cols[name] = ic.where(both.sum(axis=1) >= 10)
            self._ic = pd.DataFrame(cols)
        return self._ic

    def ic_history(self, ts: pd.Timestamp, periods: int, step: int | None = None) -> pd.DataFrame:
        """The last `periods` non-overlapping IC observations that were KNOWN at `ts`."""
        step = step or self.horizon
        ic = self.daily_ic()
        pos = ic.index.get_indexer([ts])[0]
        if pos < 0:
            raise KeyError(ts)
        last_known = pos - self.horizon  # its forward window ends exactly at ts
        if last_known < 0:
            return ic.iloc[:0]
        idx = [last_known - k * step for k in range(periods) if last_known - k * step >= 0]
        return ic.iloc[sorted(idx)]
