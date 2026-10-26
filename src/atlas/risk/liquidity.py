"""Trading-cost estimates from daily prices.

Corwin & Schultz (2012, J. Finance): the high-low range over two days contains both the
stock's volatility and its bid-ask spread; volatility scales with the length of the window
while the spread does not, so comparing one-day and two-day ranges separates them.

    beta  = ln(H_t/L_t)^2 + ln(H_{t-1}/L_{t-1})^2
    gamma = ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))^2
    alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))
    S     = 2 (e^alpha - 1) / (1 + e^alpha)        (negative estimates set to 0)

The estimate on day t uses days t-1 and t only, so it is point-in-time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling mean of daily proportional spread estimates (0.002 = 20 bps round trip)."""
    h, lo = high.astype(float), low.astype(float)
    beta = np.log(h / lo) ** 2 + np.log(h.shift(1) / lo.shift(1)) ** 2
    hh = np.maximum(h, h.shift(1))
    ll = np.minimum(lo, lo.shift(1))
    gamma = np.log(hh / ll) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = s.clip(lower=0.0)
    return s.rolling(window, min_periods=max(5, window // 2)).mean()


def abdi_ranaldo_spread(
    close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int = 63
) -> pd.DataFrame:
    """Abdi & Ranaldo (2017, RFS) close-high-low spread estimator.

        eta_t = (ln H_t + ln L_t) / 2          (mid-range: proxy for the efficient price)
        s^2   = 4 * E[(c_{t-1} - eta_{t-1}) * (c_{t-1} - eta_t)],  c = ln close

    Uses days t-1 and t, so the value on day t is point-in-time. Averaging the product over
    `window` days before taking the square root follows the paper; negative averages are
    set to 0. More accurate than Corwin-Schultz on daily data, which is inflated by
    volatility and overnight gaps.
    """
    c = np.log(close.astype(float))
    eta = (np.log(high.astype(float)) + np.log(low.astype(float))) / 2.0
    prod = (c.shift(1) - eta.shift(1)) * (c.shift(1) - eta)
    s2 = 4.0 * prod.rolling(window, min_periods=window // 2).mean()
    return np.sqrt(s2.clip(lower=0.0))


DEFAULT_TIER_COSTS = (0.0004, 0.0010, 0.0025)  # round trip: most / middle / least liquid


def liquidity_tier_cost(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    tier_costs: tuple[float, float, float] = DEFAULT_TIER_COSTS,
    window: int = 63,
) -> pd.DataFrame:
    """Round-trip cost per stock-day by liquidity tier (ASSUMED costs, not measured).

    Each day, stocks are ranked by their trailing median dollar volume (point-in-time) and
    split into thirds. Daily-bar spread estimators (Corwin-Schultz, Abdi-Ranaldo) proved
    unreliable for liquid stocks here (e.g. 25-45 bps for JPM/AAPL against real spreads of a
    few bps), so costs are explicit assumptions, reported alongside a sensitivity check.
    """
    dollar = (close * volume).rolling(window, min_periods=window // 3).median()
    pct = dollar.rank(axis=1, pct=True)
    low, mid, high = tier_costs[2], tier_costs[1], tier_costs[0]
    out = pd.DataFrame(
        np.where(pct > 2 / 3, high, np.where(pct > 1 / 3, mid, low)),
        index=close.index,
        columns=close.columns,
    )
    return out.where(dollar.notna())
