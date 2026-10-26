import numpy as np
import pandas as pd
import torch

from atlas.ml.deep_patterns import (
    IMG_H,
    IMG_W,
    ChartCNN,
    PatchTransformer,
    build_samples,
    chart_image,
    unpack_images,
)


def test_chart_image_draws_bars_in_the_right_place() -> None:
    n = 20
    up = np.linspace(10, 20, n)  # steadily rising
    img = chart_image(up, up + 0.2, up - 0.2, up, np.ones(n), up)
    assert img.shape == (IMG_H, IMG_W)
    first_col = np.nonzero(img[:51, 1])[0]
    last_col = np.nonzero(img[:51, 3 * 19 + 1])[0]
    assert first_col.min() > last_col.max()  # later (higher) prices sit nearer the top
    assert img[IMG_H - 1, 1] == 255  # volume bar touches the bottom


def _bars(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    out = {}
    for k in range(3):
        c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        out[f"EQ:S{k}"] = pd.DataFrame(
            {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=idx
        )
    return out, idx


def test_samples_use_only_past_data_and_label_the_future() -> None:
    bars, idx = _bars()
    s = build_samples(bars, idx, stride=20)
    altered = {k: v.copy() for k, v in bars.items()}
    cut = s.keys["date"].iloc[5]
    for v in altered.values():
        v.loc[v.index > cut, ["open", "high", "low", "close"]] *= 5  # change the future
    s2 = build_samples(altered, idx, stride=20)
    same = s.keys["date"] <= cut
    assert np.array_equal(s.images[same.to_numpy()], s2.images[same.to_numpy()])
    assert np.allclose(s.sequences[same.to_numpy()], s2.sequences[same.to_numpy()])
    assert np.isnan(s.target[-1])  # the last samples have no 20-day future yet
    assert set(np.unique(s.target[np.isfinite(s.target)])) <= {0.0, 1.0}
    assert unpack_images(s.images[:2]).shape == (2, 1, IMG_H, IMG_W)


def test_models_produce_two_class_outputs() -> None:
    assert ChartCNN()(torch.zeros(4, 1, IMG_H, IMG_W)).shape == (4, 2)
    assert PatchTransformer()(torch.zeros(4, 120, 2)).shape == (4, 2)
