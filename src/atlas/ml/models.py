"""Return-ranking models and purged time-series cross-validation.

Models predict each stock's centred cross-sectional return rank. They are judged by the
information coefficient (Spearman correlation per date, averaged), not by R^2: we trade
the ordering, and a model can have a tiny R^2 and still rank usefully.

* `RidgeRanker`    linear baseline: anything fancier must beat it out of sample.
* `GBMRanker`      gradient-boosted trees (histogram GBDT, LightGBM-style), strongly
                   regularised: shallow trees, large leaves, early stopping.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge


class Ranker(Protocol):
    name: str

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Ranker: ...
    def predict(self, x: pd.DataFrame) -> np.ndarray: ...


@dataclass
class RidgeRanker:
    alpha: float = 10.0
    name: str = "ridge"
    _model: Ridge | None = field(default=None, repr=False)
    _cols: list[str] = field(default_factory=list, repr=False)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> RidgeRanker:
        self._cols = list(x.columns)
        self._model = Ridge(alpha=self.alpha).fit(x.fillna(0.0), y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(x[self._cols].fillna(0.0))

    def importance(self) -> pd.Series:
        assert self._model is not None
        return pd.Series(self._model.coef_, index=self._cols)


@dataclass
class GBMRanker:
    max_leaf_nodes: int = 15
    learning_rate: float = 0.03
    max_iter: int = 400
    min_samples_leaf: int = 400  # every leaf must cover ~a full rebalance date of stocks
    l2_regularization: float = 1.0
    max_features: float = 0.7
    random_state: int = 0
    name: str = "gbm"
    _model: HistGradientBoostingRegressor | None = field(default=None, repr=False)
    _cols: list[str] = field(default_factory=list, repr=False)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> GBMRanker:
        self._cols = list(x.columns)
        self._model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_leaf_nodes=self.max_leaf_nodes,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_features=self.max_features,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=self.random_state,
        ).fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(x[self._cols])

    def importance(self, x: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.Series:
        """Permutation importance: how much the score drops when a feature is shuffled."""
        assert self._model is not None
        res = permutation_importance(
            self._model, x[self._cols], y, n_repeats=3, random_state=seed, n_jobs=1
        )
        return pd.Series(res.importances_mean, index=self._cols)


def daily_ic(pred: pd.Series, target: pd.Series) -> pd.Series:
    """Spearman IC per date between predictions and realised target ranks."""
    df = pd.DataFrame({"p": pred, "t": target}).dropna()
    out = {}
    for d, g in df.groupby(level="date"):
        if len(g) >= 10 and g["p"].nunique() > 1:
            out[d] = stats.spearmanr(g["p"], g["t"]).statistic
    return pd.Series(out, dtype=float)


def purged_time_splits(
    dates: Sequence[pd.Timestamp], label_end: pd.Series, n_splits: int = 4, min_train: int = 24
) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    """Expanding-window CV over sample dates. Training dates whose label is not yet known
    at the start of the validation block are PURGED (Lopez de Prado 2018, ch. 7)."""
    d = sorted(set(dates))
    if len(d) < min_train + n_splits:
        return []
    block = (len(d) - min_train) // n_splits
    splits = []
    for k in range(n_splits):
        val_start = min_train + k * block
        val = d[val_start : val_start + block] if k < n_splits - 1 else d[val_start:]
        cutoff = val[0]
        train = [t for t in d[:val_start] if pd.Timestamp(label_end[t]) < cutoff]
        if train and val:
            splits.append((train, val))
    return splits


def cross_validate(
    make: Callable[[], Any],
    data: pd.DataFrame,
    features: list[str],
    n_splits: int = 4,
) -> float:
    """Mean out-of-fold IC of a model configuration on the training data only."""
    label_end = data.groupby(level="date")["label_end"].first()
    splits = purged_time_splits(list(label_end.index), label_end, n_splits)
    if not splits:
        return float("nan")
    ics = []
    dates = data.index.get_level_values("date")
    for train, val in splits:
        tr, va = data[dates.isin(train)], data[dates.isin(val)]
        tr = tr[tr["target"].notna()]
        model = make().fit(tr[features], tr["target"])
        pred = pd.Series(model.predict(va[features]), index=va.index)
        ics.append(daily_ic(pred, va["target"]).mean())
    return float(np.nanmean(ics))
