import numpy as np

from atlas.ml.text_embed import NeuralHead


def test_neural_head_learns_a_planted_direction_and_early_stops() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(size=32)
    x = rng.normal(size=(6000, 32)).astype(np.float32)
    r = x @ w * 0.01 + rng.normal(0, 0.01, 6000)  # reaction depends on one direction
    head = NeuralHead(hidden=32, epochs=40, patience=3).fit(
        x[:4000], r[:4000], x[4000:5000], r[4000:5000]
    )
    s = head.score(x[5000:])
    assert np.corrcoef(s, r[5000:])[0, 1] > 0.5
    # pure noise: validation loss stops improving, so training stops early
    noise = NeuralHead(hidden=32, epochs=40, patience=3).fit(
        x[:4000], rng.normal(size=4000), x[4000:5000], rng.normal(size=1000)
    )
    assert len(noise.history) < 40
    assert abs(np.corrcoef(noise.score(x[5000:]), r[5000:])[0, 1]) < 0.15
