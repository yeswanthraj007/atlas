import numpy as np
import pandas as pd
import pytest

from atlas.risk.liquidity import corwin_schultz_spread


def simulate(spread: float, vol: float = 0.01, days: int = 400, seed: int = 0):
    """Intraday paths with a known bid-ask spread: highs print at the ask, lows at the bid."""
    rng = np.random.default_rng(seed)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, vol, days)))
    paths = mid[:, None] * np.exp(np.cumsum(rng.normal(0, vol / np.sqrt(390), (days, 390)), axis=1))
    high = paths.max(axis=1) * (1 + spread / 2)
    low = paths.min(axis=1) * (1 - spread / 2)
    idx = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame({"X": high}, index=idx), pd.DataFrame({"X": low}, index=idx)


def test_recovers_known_spreads_in_order() -> None:
    est = {
        s: corwin_schultz_spread(*simulate(s)).iloc[-200:].mean().item()
        for s in (0.0005, 0.005, 0.02)
    }
    assert est[0.0005] < est[0.005] < est[0.02]
    assert est[0.02] == pytest.approx(0.02, rel=0.5)  # right order of magnitude


def test_uses_no_future_data() -> None:
    high, low = simulate(0.01)
    full = corwin_schultz_spread(high, low)
    cut = corwin_schultz_spread(high.iloc[:250], low.iloc[:250])
    assert full.iloc[249].item() == pytest.approx(cut.iloc[249].item())
