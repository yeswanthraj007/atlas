import numpy as np
import pytest

from atlas.validation.overfit import min_detectable_ic, pbo_cscv, periods_needed

RNG = np.random.default_rng(7)


def test_pbo_is_high_when_choosing_among_pure_noise() -> None:
    noise = RNG.normal(0, 0.01, (480, 20))  # 20 useless variants
    res = pbo_cscv(noise, n_blocks=8)
    assert 0.3 < res["pbo"] < 0.7  # picking the best noise is a coin flip out of sample
    assert res["mean_sharpe_degradation"] < 0
    assert res["splits"] == 70  # C(8, 4)


def test_pbo_is_low_when_one_variant_is_genuinely_better() -> None:
    r = RNG.normal(0, 0.01, (480, 10))
    r[:, 3] += 0.004  # a real edge
    assert pbo_cscv(r, n_blocks=8)["pbo"] < 0.1


def test_power_arithmetic() -> None:
    # 21 monthly periods with IC volatility 0.15 can only confirm an IC of about 0.065
    assert min_detectable_ic(21, 0.15) == pytest.approx(0.0655, abs=1e-3)
    # an IC of 0.03 needs about 100 months (8+ years) to reach t = 2 on average
    assert periods_needed(0.03, 0.15) == 100
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 1)))
