"""Combining forecasters (pre-registered: research/preregistered/2026-11-07-forecaster-
shootout-v1.md, Amendment 2).

Members are chosen and calibrated on a SELECTION window only; everything is then scored on the
later EVALUATION window. The evaluation window never influences a choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from atlas.ml.forecasters import qlike, t_stat


def zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return (df[col] - g.transform("mean")) / (g.transform("std") + 1e-12)


def ic_series(df: pd.DataFrame, pred: pd.Series) -> pd.Series:
    d = df[["date", "fwd_ret"]].assign(p=pred)
    return d.groupby("date").apply(lambda g: stats.spearmanr(g["p"], g["fwd_ret"]).statistic)


def returns_stage(df: pd.DataFrame, models: list[str], split: pd.Timestamp, k: int = 3) -> dict:
    sel, ev = df["date"] < split, df["date"] >= split
    z = {m: zscore_by_date(df, f"ret_{m}") for m in models}
    z["ridge"] = zscore_by_date(df, "ret_ridge")
    sel_ic = {m: float(ic_series(df[sel], z[m][sel]).mean()) for m in models}
    ranked = sorted(models, key=lambda m: -sel_ic[m])
    top = ranked[:k]
    z["ENS-TOP3"] = sum(z[m] for m in top) / len(top)
    z["ENS-ALL"] = sum(z[m] for m in models) / len(models)
    ic = {name: ic_series(df[ev], zz[ev]) for name, zz in z.items()}
    best = ranked[0]
    return {
        "selected": top,
        "best_single": best,
        "selection_ic": sel_ic,
        "eval_ic": {n: float(s.mean()) for n, s in ic.items()},
        "eval_t": {n: t_stat(s) for n, s in ic.items()},
        "t_vs_ridge": {n: t_stat(ic[n] - ic["ridge"]) for n in ic if n != "ridge"},
        "t_vs_best_single": {n: t_stat(ic[n] - ic[best]) for n in ("ENS-TOP3", "ENS-ALL")},
        "eval_dates": int(ic["ridge"].shape[0]),
        "eval_rows": int(ev.sum()),
    }


def calibrate(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """OLS truth = a + b x pred; returns (a, b)."""
    b, a = np.polyfit(pred, truth, 1)
    return float(a), float(b)


def vol_stage(df: pd.DataFrame, models: list[str], split: pd.Timestamp, k: int = 3) -> dict:
    sel, ev = (df["date"] < split).to_numpy(), (df["date"] >= split).to_numpy()
    names = [*models, "har"]
    cal: dict[str, np.ndarray] = {}
    for m in names:
        col = f"vol_{m}"
        a, b = calibrate(df.loc[sel, col].to_numpy(), df.loc[sel, "fwd_log_vol"].to_numpy())
        cal[m] = a + b * df[col].to_numpy()
    truth = df["fwd_log_vol"].to_numpy()
    sel_q = {m: float(qlike(cal[m][sel], truth[sel]).mean()) for m in models}
    ranked = sorted(models, key=lambda m: sel_q[m])
    top, best = ranked[:k], ranked[0]
    cal["ENS-TOP3"] = np.mean([cal[m] for m in top], axis=0)
    cal["ENS-ALL"] = np.mean([cal[m] for m in models], axis=0)
    d = df.loc[ev, ["date"]].reset_index(drop=True)
    loss = {n: pd.Series(qlike(v[ev], truth[ev])).groupby(d["date"]).mean() for n, v in cal.items()}
    acc = {
        n: float(
            pd.DataFrame({"d": d["date"], "p": v[ev], "t": truth[ev]})
            .groupby("d")
            .apply(lambda g: ((g["p"] > g["p"].median()) == (g["t"] > g["t"].median())).mean())
            .mean()
        )
        for n, v in cal.items()
    }
    return {
        "selected": top,
        "best_single": best,
        "selection_qlike": sel_q,
        "eval_qlike": {n: float(s.mean()) for n, s in loss.items()},
        "eval_accuracy": acc,
        "dm_t_vs_har": {n: t_stat(loss["har"] - loss[n]) for n in loss if n != "har"},
        "t_vs_best_single": {n: t_stat(loss[best] - loss[n]) for n in ("ENS-TOP3", "ENS-ALL")},
        "eval_dates": len(loss["har"]),
        "eval_rows": int(ev.sum()),
    }
