import numpy as np
import pandas as pd
import torch

from atlas.ml.market_gpt import (
    CONTEXT,
    PRICE_FEATURES,
    MarketGPT,
    build_panel,
    gather,
    price_features,
    qlike,
    sample_index,
)


def _bars(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * np.exp(rng.normal(0, 0.005, n))
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) * 1.01,
            "low": np.minimum(o, c) * 0.99,
            "close": c,
            "volume": rng.integers(1_000, 5_000, n).astype(float),
        },
        index=idx,
    )


def test_price_features_use_no_future_data():
    b = _bars(300, 0)
    mkt = _bars(300, 1)["close"]
    f = price_features(b, mkt)
    b2, mkt2 = b.copy(), mkt.copy()
    b2.iloc[200:] *= 3.0
    mkt2.iloc[200:] *= 0.5
    f2 = price_features(b2, mkt2)
    pd.testing.assert_frame_equal(f.iloc[:200], f2.iloc[:200])


def test_causal_mask_hides_later_days():
    torch.manual_seed(0)
    m = MarketGPT(emb_dim=4, width=32, layers=2, heads=2).eval()
    x = torch.randn(2, CONTEXT, len(PRICE_FEATURES) + 2)
    e = torch.randn(2, CONTEXT, 4)
    h = m.hidden(x, e)
    x2, e2 = x.clone(), e.clone()
    x2[:, 80:] += 5.0
    e2[:, 80:] -= 3.0
    h2 = m.hidden(x2, e2)
    assert torch.allclose(h[:, :80], h2[:, :80], atol=1e-5)
    assert not torch.allclose(h[:, 80:], h2[:, 80:])


def test_panel_samples_and_targets():
    bars = {"EQ:A": _bars(400, 2), "EQ:B": _bars(400, 3)}
    dates = bars["EQ:A"].index
    news = {"EQ:A": (np.array([150, 151]), np.array([2, 1]), np.ones((2, 3), np.float16))}
    p = build_panel(bars, _bars(400, 4)["close"], dates, news, emb_dim=3)
    rows = sample_index(p, stride=5)
    assert (rows["end"] >= CONTEXT - 1 + 40).all()  # 40-day trailing-vol warm-up excluded
    x, e, _ = gather(p, rows["stock"].to_numpy()[:3], rows["end"].to_numpy()[:3], True)
    assert x.shape == (3, CONTEXT, len(PRICE_FEATURES) + 2)
    assert e.shape == (3, CONTEXT, 3)
    assert torch.isfinite(x).all()
    # labels: last rows have unknown 20-day outcomes
    late = rows[rows["end"] > len(dates) - 21]
    assert late["target"].isna().all() and late["label_end"].isna().all()
    # news only on its day; without news the inputs are zeroed
    assert p.news_meta[0, 150, 1] == 0 and p.news_meta[0, 149, 1] == 1
    _, e0, _ = gather(p, np.array([0]), np.array([160]), False)
    assert float(e0.abs().sum()) == 0.0


def test_qlike_is_zero_only_for_perfect_forecast():
    t = np.log(np.array([0.2, 0.3]))
    assert np.allclose(qlike(t, t), 0.0)
    assert (qlike(t + 0.1, t) > 0).all() and (qlike(t - 0.1, t) > 0).all()
