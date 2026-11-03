"""Risk and event predictors (pre-registered: research/preregistered/2026-11-03-risk-events-v1.md).

Targets that have real structure, all balanced by construction (base rate 50%):
  E1  earnings-day move is larger than the median move of that quarter's announcements
  E2  earnings-day move is larger than the stock's own median over its previous 8 events
  D1  worst drawdown over the next 21 trading days is worse than that date's median

Every feature at position p uses data up to and including the close of day p.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from atlas.ml.news_model import news_day_index
from atlas.ml.volatility import log_realized_vol

NY = "America/New_York"
TRADING_DAYS = 252

BASE_FEATURES = [
    "lrv_5", "lrv_21", "lrv_63", "lrv_252", "ret_1m", "ret_3m", "dd_252", "maxloss_21",
    "skew_63", "vol_ratio", "log_dollar_vol", "mkt_lrv_21", "days_since_event",
]  # fmt: skip
WIKI_FEATURES = ["wiki_att", "wiki_lvl", "wiki_chg"]
EVENT_FEATURES = [
    "prev1", "prev2_mean", "prev4_mean", "med8", "hour_et", "n_prior",
    "med8_over_vol63", "prev1_over_med8",
]  # fmt: skip


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------


def days_since_event_panel(index: pd.DatetimeIndex, columns: pd.Index, d0: pd.DataFrame):
    """Trading days since each stock's last earnings reaction day (d0 <= p). `d0` has the
    columns "instrument" and "d0" (position of the reaction day). NaN before the first."""
    out = np.full((len(index), len(columns)), np.nan)
    col = {c: i for i, c in enumerate(columns)}
    pos = np.arange(len(index))
    for inst, g in d0.groupby("instrument"):
        if inst not in col:
            continue
        ds = np.sort(g["d0"].to_numpy())
        k = np.searchsorted(ds, pos, side="right") - 1  # last event with d0 <= p
        ok = k >= 0
        out[ok, col[inst]] = pos[ok] - ds[k[ok]]
    return pd.DataFrame(out, index=index, columns=columns)


def daily_features(
    closes: pd.DataFrame,
    volume: pd.DataFrame,
    split_closes: pd.DataFrame,
    benchmark: pd.Series,
    dse: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Feature panels as float32 arrays [date, stock]."""
    r = closes.pct_change(fill_method=None)
    mkt = np.log(
        benchmark.reindex(closes.index).pct_change().rolling(21).std() * np.sqrt(TRADING_DAYS)
    )
    vr = np.log(
        volume.rolling(5, min_periods=3).mean() / volume.rolling(60, min_periods=40).mean()
    ).replace([np.inf, -np.inf], np.nan)
    panels = {
        "lrv_5": log_realized_vol(closes, 5),
        "lrv_21": log_realized_vol(closes, 21),
        "lrv_63": log_realized_vol(closes, 63),
        "lrv_252": log_realized_vol(closes, 252),
        "ret_1m": closes / closes.shift(21) - 1.0,
        "ret_3m": closes / closes.shift(63) - 1.0,
        "dd_252": closes / closes.rolling(252, min_periods=126).max() - 1.0,
        "maxloss_21": r.rolling(21, min_periods=15).min(),
        "skew_63": r.rolling(63, min_periods=40).skew(),
        "vol_ratio": vr,
        "log_dollar_vol": np.log(
            (split_closes * volume).rolling(60, min_periods=40).mean().where(lambda x: x > 0)
        ),
        "days_since_event": dse,
    }
    out = {k: v.to_numpy(np.float32) for k, v in panels.items()}
    out["mkt_lrv_21"] = np.repeat(mkt.to_numpy(np.float32)[:, None], closes.shape[1], axis=1)
    return out


def wiki_features(views: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index):
    """Attention features from daily page views (calendar days, index tz-aware UTC, columns
    are tickers). Lagged one calendar day: the value on date t uses views through t-1."""
    v = views.copy()
    v.index = pd.DatetimeIndex(v.index).tz_convert("UTC").tz_localize(None).normalize()
    full = pd.date_range(v.index.min(), v.index.max(), freq="D")
    v = v.reindex(full).ffill(limit=3).shift(1)
    m7, m90 = v.rolling(7, min_periods=5).mean(), v.rolling(90, min_periods=60).mean()
    prev7 = v.shift(7).rolling(7, min_periods=5).mean()
    feats = {
        "wiki_att": np.log((m7 + 1) / (m90 + 1)),
        "wiki_lvl": np.log(m90 + 1),
        "wiki_chg": np.log(m7 + 1) - np.log(prev7 + 1),
    }
    dates = (index + pd.Timedelta(hours=12)).tz_convert("UTC").tz_localize(None).normalize()
    out = {}
    for k, f in feats.items():
        f = f.reindex(dates)
        f.index = index
        f.columns = ["EQ:" + c for c in f.columns]
        out[k] = f.reindex(columns=columns).to_numpy(np.float32)
    return out


# --------------------------------------------------------------------------------------
# Earnings events (E1, E2)
# --------------------------------------------------------------------------------------


def build_events(
    closes: pd.DataFrame, benchmark: pd.Series, earnings: pd.DataFrame
) -> pd.DataFrame:
    """One row per earnings announcement. `earnings` has symbol and accepted (UTC time).

    d0 is the first trading day whose 16:00 ET close is at or after the announcement. The
    reaction is close(d0) / close(d0 - 1) minus the benchmark's, so an after-hours
    release is measured on the next session and a pre-open release on the same day."""
    e = earnings.copy()
    e["accepted"] = pd.to_datetime(e["accepted"], utc=True)
    e["instrument"] = "EQ:" + e["symbol"].astype(str)
    e = e[e["instrument"].isin(closes.columns)]
    e["day"] = e["accepted"].dt.tz_convert(NY).dt.normalize()
    e = e.sort_values("accepted").drop_duplicates(["instrument", "day"])
    d0 = news_day_index(e["accepted"], closes.index)
    e = e.assign(d0=d0)
    e = e[(e["d0"] >= 1)].reset_index(drop=True)
    col = {c: i for i, c in enumerate(closes.columns)}
    j = e["instrument"].map(col).to_numpy()
    c = closes.to_numpy()
    b = benchmark.reindex(closes.index).to_numpy()
    d = e["d0"].to_numpy()
    e["react"] = (c[d, j] / c[d - 1, j] - 1.0) - (b[d] / b[d - 1] - 1.0)
    e["abs_react"] = e["react"].abs()
    e["p"] = e["d0"] - 1
    e["stock"] = j
    e["date"] = closes.index[d]
    local = e["accepted"].dt.tz_convert(NY)
    e["hour_et"] = local.dt.hour + local.dt.minute / 60.0
    e = e.sort_values(["instrument", "accepted"]).reset_index(drop=True)
    g = e.groupby("instrument")["abs_react"]
    e["prev1"] = g.shift(1)
    e["prev2_mean"] = g.transform(lambda s: s.rolling(2, min_periods=1).mean().shift(1))
    e["prev4_mean"] = g.transform(lambda s: s.rolling(4, min_periods=2).mean().shift(1))
    e["med8"] = g.transform(lambda s: s.rolling(8, min_periods=4).median().shift(1))
    e["n_prior"] = e.groupby("instrument").cumcount()
    q = e["accepted"].dt.year * 10 + e["accepted"].dt.quarter
    e["y_e1"] = np.where(
        e["abs_react"].notna(),
        (e["abs_react"] > e.groupby(q)["abs_react"].transform("median")).astype(float),
        np.nan,
    )
    e["y_e2"] = np.where(
        e["abs_react"].notna() & e["med8"].notna(),
        (e["abs_react"] > e["med8"]).astype(float),
        np.nan,
    )
    return e.sort_values("accepted").reset_index(drop=True)


def add_event_ratios(e: pd.DataFrame, lrv_63: np.ndarray) -> pd.DataFrame:
    daily_vol = np.exp(lrv_63[e["p"].to_numpy(), e["stock"].to_numpy()]) / np.sqrt(TRADING_DAYS)
    e = e.copy()
    e["med8_over_vol63"] = e["med8"] / daily_vol
    e["prev1_over_med8"] = e["prev1"] / e["med8"]
    return e


# --------------------------------------------------------------------------------------
# Drawdown samples (D1)
# --------------------------------------------------------------------------------------


def build_drawdown_samples(
    closes: pd.DataFrame, membership: pd.DataFrame, step: int = 21, start: int = 260, h: int = 21
) -> pd.DataFrame:
    """Every `step`-th trading day: worst drawdown over the next `h` days from the close."""
    fwd_min = closes.rolling(h, min_periods=h).min().shift(-h)
    fwd_dd = (fwd_min / closes - 1.0).to_numpy()
    mem = membership.reindex(index=closes.index, columns=closes.columns).fillna(False).to_numpy()
    rows = []
    for p in range(start, len(closes) - h, step):
        ok = mem[p] & np.isfinite(fwd_dd[p]) & closes.iloc[p].notna().to_numpy()
        j = np.where(ok)[0]
        rows.append(pd.DataFrame({"p": p, "stock": j, "fwd_dd": fwd_dd[p, j]}))
    s = pd.concat(rows, ignore_index=True)
    s["instrument"] = closes.columns[s["stock"].to_numpy()]
    s["date"] = closes.index[s["p"].to_numpy()]
    s["label_date"] = closes.index[np.minimum(s["p"].to_numpy() + h, len(closes) - 1)]
    s["y_d1"] = (s["fwd_dd"] < s.groupby("p")["fwd_dd"].transform("median")).astype(float)
    return s


def take(panels: dict[str, np.ndarray], names: list[str], p: np.ndarray, j: np.ndarray):
    return pd.DataFrame({n: panels[n][p, j] for n in names})


# --------------------------------------------------------------------------------------
# Models, baselines, scoring
# --------------------------------------------------------------------------------------

GRID = [(3, 200), (3, 400), (5, 200), (5, 400)]


def fit_hgb(
    x: pd.DataFrame, y: np.ndarray, depth: int, iters: int
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=depth, max_iter=iters, random_state=0,
        early_stopping=False,
    ).fit(x, y)  # fmt: skip


def choose_and_fit(x_tr, y_tr, x_va, y_va, x_all, y_all):
    """Hyperparameters by validation log loss (train fit); the final model is refitted on
    train + validation with the chosen setting."""
    scores = {}
    for depth, iters in GRID:
        m = fit_hgb(x_tr, y_tr, depth, iters)
        scores[(depth, iters)] = log_loss(y_va, m.predict_proba(x_va)[:, 1])
    best = min(scores, key=scores.get)
    return fit_hgb(x_all, y_all, *best), best, scores


def predict_above_median(score: np.ndarray, group: np.ndarray) -> np.ndarray:
    """1 if the score is above its group's median (ties go to 0)."""
    s = pd.Series(score)
    return (s > s.groupby(group).transform("median")).to_numpy().astype(float)


def month_accuracy(pred: np.ndarray, y: np.ndarray, month: np.ndarray) -> pd.Series:
    ok = np.isfinite(y)
    df = pd.DataFrame({"hit": (pred == y)[ok], "m": month[ok]})
    return df.groupby("m")["hit"].mean()


def paired_t(a: pd.Series, b: pd.Series) -> float:
    d = (a - b).dropna()
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 else float("nan")


@dataclass
class TaskResult:
    name: str
    accuracy: float
    auc: float
    confident_accuracy: float
    best_baseline: str
    baseline_accuracy: dict[str, float]
    margin: float
    margin_t: float
    months: int
    n: int
    monthly: pd.Series


def confident_accuracy(prob: np.ndarray, y: np.ndarray, month: np.ndarray, q: float = 0.2):
    df = pd.DataFrame({"p": prob, "y": y, "m": month}).dropna(subset=["y"])
    rk = df.groupby("m")["p"].rank(pct=True)
    sel = (rk >= 1 - q) | (rk <= q)
    pred = (rk[sel] >= 1 - q).astype(float)
    return float((pred == df.loc[sel, "y"]).mean()) if sel.any() else float("nan")


def score_task(
    name: str, prob: np.ndarray, y: np.ndarray, month: np.ndarray, baselines: dict[str, np.ndarray]
) -> TaskResult:
    ok = np.isfinite(y)
    pred = (prob > 0.5).astype(float)
    acc_m = month_accuracy(pred, y, month)
    base_m = {k: month_accuracy(v, y, month) for k, v in baselines.items()}
    base_acc = {k: float((v == y)[ok].mean()) for k, v in baselines.items()}
    best = max(base_acc, key=base_acc.get)
    return TaskResult(
        name=name,
        accuracy=float((pred == y)[ok].mean()),
        auc=float(roc_auc_score(y[ok], prob[ok])),
        confident_accuracy=confident_accuracy(prob, y, month),
        best_baseline=best,
        baseline_accuracy=base_acc,
        margin=float((pred == y)[ok].mean() - base_acc[best]),
        margin_t=paired_t(acc_m, base_m[best]),
        months=len(acc_m),
        n=int(ok.sum()),
        monthly=acc_m,
    )
