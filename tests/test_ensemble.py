import numpy as np
import pandas as pd

from atlas.ml.ensemble import calibrate, returns_stage, vol_stage, zscore_by_date

MODELS = ["a", "b", "c", "d", "e"]
SPLIT = pd.Timestamp("2025-01-01")


def _frame(seed=0, n_dates=24, n_stocks=300):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-31", periods=n_dates, freq="ME")
    d = np.repeat(dates, n_stocks)
    n = len(d)
    signal = rng.normal(size=n)
    truth_vol = rng.normal(-1.3, 0.4, n)
    df = pd.DataFrame({"date": d, "instrument": np.tile(np.arange(n_stocks), n_dates)})
    df["fwd_ret"] = 0.3 * signal + rng.normal(size=n)  # IC about 0.29 for a perfect read
    df["fwd_log_vol"] = truth_vol
    # models a, b: noisy reads of the signal; c: weak; d, e: pure noise
    for m, noise in zip(MODELS, [1.0, 1.5, 6.0, None, None], strict=True):
        df[f"ret_{m}"] = signal + noise * rng.normal(size=n) if noise else rng.normal(size=n)
        df[f"vol_{m}"] = truth_vol + (0.2 if noise else 1.0) * rng.normal(size=n) + 0.5
    df["ret_ridge"] = rng.normal(size=n)
    df["vol_har"] = truth_vol + 0.3 * rng.normal(size=n)
    return df


def test_selection_picks_the_informative_models_and_averaging_helps():
    r = returns_stage(_frame(), MODELS, SPLIT)
    assert set(r["selected"][:2]) == {"a", "b"}
    assert r["best_single"] == "a"
    assert r["eval_ic"]["ENS-TOP3"] > 0.8 * r["eval_ic"]["a"]  # not hurt by the noise model
    assert r["eval_ic"]["ENS-ALL"] < r["eval_ic"]["ENS-TOP3"]  # averaging in noise dilutes it
    assert r["eval_ic"]["ridge"] < 0.05


def test_choices_never_use_the_evaluation_window():
    df = _frame(seed=1)
    base = returns_stage(df, MODELS, SPLIT)
    ev = df["date"] >= SPLIT
    df2 = df.copy()
    rng = np.random.default_rng(9)
    for m in MODELS:  # scramble every model's evaluation-window forecasts
        df2.loc[ev, f"ret_{m}"] = rng.normal(size=int(ev.sum()))
        df2.loc[ev, f"vol_{m}"] = rng.normal(size=int(ev.sum()))
    again = returns_stage(df2, MODELS, SPLIT)
    assert again["selected"] == base["selected"] and again["selection_ic"] == base["selection_ic"]
    v1, v2 = vol_stage(df, MODELS, SPLIT), vol_stage(df2, MODELS, SPLIT)
    assert v1["selected"] == v2["selected"] and v1["selection_qlike"] == v2["selection_qlike"]


def test_vol_calibration_removes_a_constant_bias():
    a, b = calibrate(np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 2.5]))
    assert abs(a + 0.5) < 1e-9 and abs(b - 1.0) < 1e-9
    v = vol_stage(_frame(seed=2), MODELS, SPLIT)
    assert v["selected"][:2] != ["d", "e"]  # the noisy models are not the best
    assert v["eval_qlike"]["a"] < v["eval_qlike"]["d"]
    assert v["eval_rows"] > 0 and v["eval_dates"] == 12


def test_zscore_is_within_date():
    df = pd.DataFrame({"date": ["x", "x", "y", "y"], "v": [1.0, 3.0, 10.0, 30.0]})
    z = zscore_by_date(df, "v")
    assert np.allclose(z.to_numpy(), [-0.7071, 0.7071, -0.7071, 0.7071], atol=1e-3)
