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
