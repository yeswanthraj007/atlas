"""Scoring the forward test (pre-registered metrics). Outcomes use only prices after the release."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from atlas.llm_forward.baseline import clip_prob
from atlas.ml.news_model import news_day_index

H = 20


def compute_outcomes(
    accepted: pd.Series, instrument: pd.Series, closes: pd.DataFrame, benchmark: pd.Series
) -> pd.DataFrame:
    """Market-adjusted reaction (first session after the release) and 20-day drift from that
    session's close. NaN where the price data does not reach yet."""
    idx = closes.index
    d0 = news_day_index(pd.to_datetime(accepted, utc=True), idx)
    col = {c: i for i, c in enumerate(closes.columns)}
    j = instrument.map(col).to_numpy()
    c = closes.to_numpy()
    b = benchmark.reindex(idx).to_numpy()
    n = len(d0)
    react, drift = np.full(n, np.nan), np.full(n, np.nan)
    ok = (d0 >= 1) & ~pd.isna(j)
    jj = np.where(ok, j, 0).astype(int)
    i = np.where(ok)[0]
    react[i] = (c[d0[i], jj[i]] / c[d0[i] - 1, jj[i]] - 1.0) - (b[d0[i]] / b[d0[i] - 1] - 1.0)
    far = ok & (d0 + H < len(idx))
    i = np.where(far)[0]
    drift[i] = (c[d0[i] + H, jj[i]] / c[d0[i], jj[i]] - 1.0) - (b[d0[i] + H] / b[d0[i]] - 1.0)
    return pd.DataFrame({"d0": d0, "react": react, "drift": drift}, index=accepted.index)


def log_loss_each(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = clip_prob(np.asarray(p, float))
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_t(diff: np.ndarray, cluster: np.ndarray) -> float:
    """t-statistic of the mean paired difference with standard errors clustered by day."""
    g = pd.Series(diff).groupby(cluster).agg(["sum", "count"])
    if len(g) < 3:
        return float("nan")
    n = g["count"].sum()
    mean = diff.mean()
    # cluster-robust variance of the mean: sum of squared cluster-total deviations / n^2
    resid = g["sum"] - mean * g["count"]
    se = np.sqrt((resid**2).sum()) / n
    return float(mean / se) if se > 0 else float("nan")


def binary_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.asarray(p, float)
    return {
        "log_loss": float(log_loss_each(p, y).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        "accuracy": float(((p > 0.5) == (y > 0.5)).mean()),
    }


def score_stream(df: pd.DataFrame, qmedian: dict[int, float], base_rates: dict[str, float]) -> dict:
    """df: one row per eligible prediction with p_reaction_up, p_drift_up, p_big_move, tfidf_t1,
    tfidf_t2, tfidf_t3, react, drift, quarter, month, day and cost (round-trip fraction).
    Rows without outcomes are ignored. Returns {} when nothing is scoreable yet."""
    d = df[df["react"].notna()].copy()
    if d.empty:
        return {}
    d["y1"] = (d["react"] > 0).astype(float)
    d["y3"] = (d["react"].abs() > d["quarter"].map(qmedian)).astype(float)
    out: dict = {"n_reaction": len(d)}
    for tag, col, base_col, y in (
        ("t1", "p_reaction_up", "tfidf_t1", "y1"),
        ("t3", "p_big_move", "tfidf_t3", "y3"),
    ):
        yy = d[y].to_numpy()
        llm = binary_metrics(d[col].to_numpy(), yy)
        tf = binary_metrics(d[base_col].to_numpy(), yy)
        const = binary_metrics(np.full(len(d), base_rates[tag]), yy)
        diff = log_loss_each(d[col].to_numpy(), yy) - log_loss_each(d[base_col].to_numpy(), yy)
        out[tag] = {
            "llm": llm, "tfidf": tf, "base_rate": const,
            "paired_t_logloss_llm_minus_tfidf": cluster_t(diff, d["day"].to_numpy()),
        }  # fmt: skip
    m = d[d["drift"].notna()]
    if len(m) >= 30:
        ic = (
            m.groupby("month")
            .apply(
                lambda g: (
                    stats.spearmanr(g["p_drift_up"], g["drift"]).statistic
                    if len(g) >= 30
                    else np.nan
                )
            )
            .dropna()
        )
        out["t2"] = {
            "n": len(m),
            "ic_all": float(stats.spearmanr(m["p_drift_up"], m["drift"]).statistic),
            "ic_tfidf_all": float(stats.spearmanr(m["tfidf_t2"], m["drift"]).statistic),
            "ic_monthly_mean": float(ic.mean()) if len(ic) else float("nan"),
            "ic_monthly_t": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))
            if len(ic) > 2
            else float("nan"),
            "months": len(ic),
        }
        k = max(3, len(m) // 10)
        s = m.sort_values("p_drift_up")
        lo, hi = s.iloc[:k], s.iloc[-k:]
        gross = hi["drift"].mean() - lo["drift"].mean()
        out["t2"]["spread_gross"] = float(gross)
        out["t2"]["spread_net"] = float(gross - hi["cost"].mean() - lo["cost"].mean())
    # calibration of p_big_move and p_reaction_up in five bins
    cal = {}
    for col, y in (("p_reaction_up", "y1"), ("p_big_move", "y3")):
        bins = pd.cut(d[col], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        cal[col] = (
            d.groupby(bins, observed=True)
            .agg(n=(y, "size"), predicted=(col, "mean"), actual=(y, "mean"))
            .round(3)
            .to_dict("index")
        )
    out["calibration"] = {k: {str(a): b for a, b in v.items()} for k, v in cal.items()}
    return out
