# ruff: noqa: E501
import numpy as np
import pandas as pd
import torch
from transformers import RobertaConfig, RobertaModel

from atlas.ml.llm_reader import (
    MAX_LEN,
    ReaderConfig,
    ReaderModel,
    drift_labels,
    gaussian_rank_by_group,
    mask_company,
    predict_reader,
    prepare_text,
    reader_loss,
    strip_header,
    train_reader,
)
from atlas.ml.text_embed import device


def test_header_is_stripped_and_company_is_masked():
    raw = "EX-99.1\n2\nq4fy26pr.htm\nEX-99.1\nNVIDIA Announces Financial Results for Fourth Quarter\nNVIDIA (NASDAQ: NVDA) reported record revenue; nvidia said NVDA rose."
    body = strip_header(raw)
    assert body.startswith("NVIDIA Announces")
    out = mask_company(body, ["NVIDIA Corporation", "NVIDIA"], "NVDA")
    assert "NVIDIA" not in out and "nvidia" not in out and "NVDA" not in out
    assert out.count("[COMPANY]") >= 4
    assert "record revenue" in out
    # a ticker inside another word is left alone
    assert mask_company("The NVDAX fund", [], "NVDA") == "The NVDAX fund"
    assert len(prepare_text("x" * 40 + "\n" + "word " * 2000, [], "")) <= 2600


def test_drift_uses_only_prices_after_the_reaction_and_is_unknown_when_too_recent():
    idx = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
    c = pd.DataFrame({"EQ:A": np.arange(100.0, 160.0)}, index=idx)
    spy = pd.Series(np.full(60, 100.0), index=idx)
    d0, stock = np.array([10, 50]), np.array([0, 0])
    out = drift_labels(c, spy, d0, stock, h=20)
    assert np.isclose(out[0], c.iloc[30, 0] / c.iloc[10, 0] - 1.0) and np.isnan(out[1])
    c2 = c.copy()
    c2.iloc[:10] *= 7.0  # changing prices BEFORE the reaction day cannot change the drift
    assert np.isclose(drift_labels(c2, spy, d0, stock, h=20)[0], out[0])


def test_gaussian_rank_is_within_group():
    v = pd.Series([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    g = pd.Series(["a", "a", "a", "b", "b", "b"])
    r = gaussian_rank_by_group(v, g)
    assert np.allclose(r[:3], r[3:]) and abs(r[1]) < 1e-6


def _tiny(n=96, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, 100, (n, MAX_LEN)).astype(np.int32)
    y = (ids[:, 0] > 52).astype(np.float32)  # a learnable rule: the first token decides
    return {
        "ids": ids,
        "mask": np.ones_like(ids, dtype=np.int8),
        "y1": y,
        "y2": y * 2 - 1,
        "y3": 1 - y,
    }


def _model():
    cfg = RobertaConfig(vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=MAX_LEN + 4)  # fmt: skip
    return ReaderModel(RobertaModel(cfg), width=32)


def test_model_shapes_loss_and_training_resume(tmp_path):
    train, val = _tiny(96, 0), _tiny(32, 1)
    m = _model()
    out = predict_reader(m, val, batch=16)
    assert out.shape == (32, 3)
    ids = torch.from_numpy(val["ids"][:4].astype(np.int64)).to(device())
    mask = torch.from_numpy(val["mask"][:4].astype(np.int64)).to(device())
    assert m(ids, mask).shape == (4, 3)
    cfg = ReaderConfig(lr=2e-3, batch=16, epochs=3)
    m = train_reader(m, train, val, cfg, tmp_path, progress=lambda _: None)
    assert [h[0] for h in cfg.history] == [0, 1, 2]
    assert cfg.history[-1][2] < cfg.history[0][2]  # validation loss improved
    # resuming a finished run does not retrain
    cfg2 = ReaderConfig(lr=2e-3, batch=16, epochs=3)
    train_reader(_model(), train, val, cfg2, tmp_path, progress=lambda _: None)
    assert [h[0] for h in cfg2.history] == [0, 1, 2]
    y = [torch.from_numpy(val[k]) for k in ("y1", "y2", "y3")]
    assert float(reader_loss(torch.from_numpy(out), *y)) > 0
