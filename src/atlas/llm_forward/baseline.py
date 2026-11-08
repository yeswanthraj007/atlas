"""The word-count (TF-IDF) baseline that rides along with every LLM prediction. Fitted once on
2014-2022 releases (the language-model-reader training split) and then frozen."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

PATH = Path("data/llm_forward/baseline.joblib")


def fit_baseline(
    dataset: str | Path = "data/llm_reader/dataset.parquet", out: str | Path = PATH
) -> dict:
    ds = pd.read_parquet(dataset, columns=["text", "split", "y1", "y2", "y3"])
    tr = ds[ds["split"] == "train"]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, sublinear_tf=True, min_df=3)
    x = vec.fit_transform(tr["text"])
    bundle = {
        "vec": vec,
        "t1": LogisticRegression(C=1.0, max_iter=1000).fit(x, tr["y1"]),
        "t3": LogisticRegression(C=1.0, max_iter=1000).fit(x, tr["y3"]),
        "t2": Ridge(alpha=1.0).fit(x, tr["y2"]),
        "base_rates": {"t1": float(tr["y1"].mean()), "t3": float(tr["y3"].mean())},
        "n_train": len(tr),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    return bundle


def load_baseline(path: str | Path = PATH) -> dict:
    return joblib.load(path)


def score_baseline(texts: list[str], bundle: dict) -> pd.DataFrame:
    """tfidf_t1 and tfidf_t3 are probabilities; tfidf_t2 is a drift score (higher = more up)."""
    x = bundle["vec"].transform(texts)
    return pd.DataFrame(
        {
            "tfidf_t1": bundle["t1"].predict_proba(x)[:, 1],
            "tfidf_t2": bundle["t2"].predict(x),
            "tfidf_t3": bundle["t3"].predict_proba(x)[:, 1],
        }
    )


def base_rate_probs(bundle: dict) -> dict[str, float]:
    return bundle["base_rates"]


def clip_prob(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return np.clip(p, eps, 1 - eps)
