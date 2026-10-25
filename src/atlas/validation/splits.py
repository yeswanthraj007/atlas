"""Time-series train/test splits for strategy validation.

Unlike ML on i.i.d. data, market data is ordered and autocorrelated, so:
  * test windows always come AFTER their training window (no shuffling);
  * an embargo gap separates them, because a decision near the end of training is
    evaluated on returns that overlap the start of the test window (Lopez de Prado 2018);
  * the most recent period is a locked holdout, evaluated once, at the very end.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"fold {self.number}: train {self.train_start.date()}..{self.train_end.date()}"
            f" | test {self.test_start.date()}..{self.test_end.date()}"
        )


@dataclass(frozen=True)
class SplitPlan:
    folds: list[Fold]
    holdout_start: pd.Timestamp | None
    holdout_end: pd.Timestamp | None


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    first_train_start: pd.Timestamp,
    min_train_days: int = 756,  # ~3 years
    test_days: int = 252,  # ~1 year
    embargo_days: int = 21,
    holdout_days: int = 252,
    expanding: bool = True,
) -> SplitPlan:
    """Expanding (or rolling) walk-forward folds on trading days, plus a final holdout.

    Every test day appears in exactly one fold; no test day is ever in a training window
    or in the holdout.
    """
    d = pd.DatetimeIndex(dates).sort_values().unique()
    d = d[d >= first_train_start]
    if holdout_days:
        if len(d) <= holdout_days + min_train_days + embargo_days:
            raise ValueError("not enough history for the requested train/test/holdout sizes")
        holdout, d_dev = d[-holdout_days:], d[: -holdout_days - embargo_days]
    else:
        holdout, d_dev = d[:0], d
    folds: list[Fold] = []
    test_start_i = min_train_days + embargo_days
    while test_start_i < len(d_dev):
        test_end_i = min(test_start_i + test_days - 1, len(d_dev) - 1)
        train_end_i = test_start_i - 1 - embargo_days
        train_start_i = 0 if expanding else max(0, train_end_i + 1 - min_train_days)
        folds.append(
            Fold(
                len(folds) + 1,
                d_dev[train_start_i],
                d_dev[train_end_i],
                d_dev[test_start_i],
                d_dev[test_end_i],
            )
        )
        test_start_i = test_end_i + 1  # test windows are back to back
    if not folds:
        raise ValueError("no folds fit in the available history")
    return SplitPlan(
        folds,
        holdout[0] if len(holdout) else None,
        holdout[-1] if len(holdout) else None,
    )
