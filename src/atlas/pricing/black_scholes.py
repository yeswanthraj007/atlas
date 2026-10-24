"""Black-Scholes-Merton pricing and Greeks for European options (vectorised).

This is the fast path used for intraday risk and Monte Carlo revaluation of large books.
American exercise, discrete dividends and vol surfaces go through QuantLib in a later
milestone; this module stays the reference those engines are checked against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

Arr = NDArray[np.float64]


@dataclass(frozen=True)
class Greeks:
    price: Arr
    delta: Arr
    gamma: Arr
    vega: Arr  # per 1.00 change in vol (divide by 100 for per vol point)
    theta: Arr  # per year
    rho: Arr


def _d1_d2(s: Arr, k: Arr, t: Arr, r: Arr, q: Arr, vol: Arr) -> tuple[Arr, Arr]:
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * vol**2) * t) / (vol * sqrt_t)
    return d1, d1 - vol * sqrt_t


def bsm(
    spot: ArrayLike,
    strike: ArrayLike,
    t: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike = 0.0,
    q: ArrayLike = 0.0,
    is_call: ArrayLike = True,
) -> Greeks:
    s, k, tt, v, rr, qq = (np.asarray(a, dtype=np.float64) for a in (spot, strike, t, vol, r, q))
    call = np.asarray(is_call, dtype=bool)
    tt = np.maximum(tt, 1e-10)  # at expiry: intrinsic value, delta becomes a step
    v = np.maximum(v, 1e-10)
    d1, d2 = _d1_d2(s, k, tt, rr, qq, v)
    df_r, df_q = np.exp(-rr * tt), np.exp(-qq * tt)
    n = stats.norm
    pdf_d1 = n.pdf(d1)

    call_px = s * df_q * n.cdf(d1) - k * df_r * n.cdf(d2)
    put_px = k * df_r * n.cdf(-d2) - s * df_q * n.cdf(-d1)
    price = np.where(call, call_px, put_px)
    delta = np.where(call, df_q * n.cdf(d1), df_q * (n.cdf(d1) - 1.0))
    gamma = df_q * pdf_d1 / (s * v * np.sqrt(tt))
    vega = s * df_q * pdf_d1 * np.sqrt(tt)
    common = -s * df_q * pdf_d1 * v / (2.0 * np.sqrt(tt))
    theta = np.where(
        call,
        common - rr * k * df_r * n.cdf(d2) + qq * s * df_q * n.cdf(d1),
        common + rr * k * df_r * n.cdf(-d2) - qq * s * df_q * n.cdf(-d1),
    )
    rho = np.where(call, k * tt * df_r * n.cdf(d2), -k * tt * df_r * n.cdf(-d2))
    return Greeks(price, delta, gamma, vega, theta, rho)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t: float,
    r: float = 0.0,
    q: float = 0.0,
    is_call: bool = True,
) -> float:
    """Brent root-find on [1e-4, 5]. Raises ValueError if the price violates no-arbitrage bounds."""
    df_r, df_q = np.exp(-r * t), np.exp(-q * t)
    lower = (
        max(spot * df_q - strike * df_r, 0.0) if is_call else max(strike * df_r - spot * df_q, 0.0)
    )
    upper = spot * df_q if is_call else strike * df_r
    if not lower - 1e-12 <= price <= upper + 1e-12:
        raise ValueError(f"price {price} outside no-arbitrage bounds [{lower:.6f}, {upper:.6f}]")

    def f(v: float) -> float:
        return float(bsm(spot, strike, t, v, r, q, is_call).price) - price

    return float(optimize.brentq(f, 1e-4, 5.0, xtol=1e-10))
