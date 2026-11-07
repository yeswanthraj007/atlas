"""Re-score the shootout after fixing the ARIMA-GARCH implementation bug.

The bug (found after the first results were seen): stocks that began trading after the
fitting cut-off (for example SOLV and GEV, spun off in 2024) had their missing history padded
with zeros, so ARIMA-GARCH was fitted to a flat line and forecast near-zero volatility.
The fix fits only on a stock's real returns and needs at least 500 real days; stocks that
fail that rule drop out of the evaluation set for EVERY model, so the comparison stays
like for like. All other models' saved predictions are reused unchanged. The first run's
trials stay in the log; this run adds corrected ones.

    uv run python scripts/forecaster_shootout_v1_fix.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from atlas.lab.trials import log_external_trial
from atlas.ml.forecasters import (
    arma_garch_for_stock,
    score_return_task,
    score_vol_task,
    t_stat,
)
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

OUT = Path("data/reports/shootout_v1")
PREREG = "research/preregistered/2026-11-07-forecaster-shootout-v1.md"
MODELS = ["arima_garch", "lstm", "marketgpt", "chronos", "timesfm"]


def _one(r: np.ndarray, cut: int, positions: np.ndarray):
    import warnings

    warnings.simplefilter("ignore")
    return arma_garch_for_stock(r, cut, positions)


def main() -> None:
    df = pd.read_parquet(OUT / "predictions.parquet")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    rd = load_research_data("price", universe="sp500")
    idx, closes = rd.closes.index, rd.closes
    p = idx.get_indexer(df["date"])
    j = closes.columns.get_indexer(df["instrument"])
    assert (p >= 0).all() and (j >= 0).all()
    cut = int(idx.searchsorted(pd.Timestamp("2024-01-02", tz="UTC"))) - 1
    logret = (100 * np.log(closes / closes.shift(1))).to_numpy()
    t0 = time.time()
    stocks = np.unique(j)
    groups = {s: np.where(j == s)[0] for s in stocks}
    res = Parallel(n_jobs=8)(delayed(_one)(logret[:, s], cut, p[groups[s]]) for s in stocks)
    ret = np.full(len(df), np.nan)
    vol = np.full(len(df), np.nan)
    for s, r in zip(stocks, res, strict=True):
        if r is not None:
            ret[groups[s]], vol[groups[s]] = r[0], np.log(r[1])
    print(f"refitted {len(stocks)} stocks in {time.time() - t0:.0f}s; "
          f"{int(np.isfinite(vol).sum()):,}/{len(df):,} rows have a valid fit")  # fmt: skip
    fixed = df.assign(ret_arima_garch=ret, vol_arima_garch=vol)
    keep = np.isfinite(fixed[[f"vol_{m}" for m in MODELS] + ["vol_har", "ret_arima_garch"]]).all(1)
    dropped = fixed.loc[~keep, "instrument"].nunique()
    fixed = fixed[keep].reset_index(drop=True)
    fixed.to_parquet(OUT / "predictions_fixed.parquet")
    print(f"evaluation set: {len(fixed):,} (date, stock) pairs; {dropped} stocks dropped "
          f"(fewer than 500 real days before 2024)")  # fmt: skip

    rets = {m: score_return_task(fixed, fixed[f"ret_{m}"].to_numpy()) for m in MODELS}
    rets["ridge (baseline)"] = score_return_task(fixed, fixed["ret_ridge"].to_numpy())
    har = fixed["vol_har"].to_numpy()
    vols = {m: score_vol_task(fixed, fixed[f"vol_{m}"].to_numpy(), har) for m in MODELS}
    vols["HAR (baseline)"] = score_vol_task(fixed, har, har)
    vols["naive (baseline)"] = score_vol_task(fixed, fixed["vol_naive"].to_numpy(), har)
    ridge_ic = rets["ridge (baseline)"]["ic_by_date"]
    for m in MODELS:
        rets[m]["t_vs_ridge"] = t_stat(rets[m]["ic_by_date"] - ridge_ic)
    rank_r = pd.Series({m: rets[m]["ic"] for m in MODELS}).rank(ascending=False)
    rank_v = pd.Series({m: vols[m]["qlike"] for m in MODELS}).rank(ascending=True)
    overall = ((rank_r + rank_v) / 2).sort_values()

    print(f"\n=== Task R (ridge baseline IC {rets['ridge (baseline)']['ic']:+.4f}) ===")
    for m in [*MODELS, "ridge (baseline)"]:
        r = rets[m]
        extra = f" | t vs ridge {r['t_vs_ridge']:+.2f}" if m in MODELS else ""
        print(
            f"  {m:18s} IC {r['ic']:+.4f} (t {r['ic_t']:+.2f}) "
            f"| accuracy {r['accuracy']:.1%}{extra}"
        )
    print("\n=== Task V (QLIKE, lower is better) ===")
    for m in [*MODELS, "HAR (baseline)", "naive (baseline)"]:
        v = vols[m]
        print(f"  {m:18s} QLIKE {v['qlike']:.4f} | DM t vs HAR {v['dm_t_vs_har']:+.2f} | "
              f"above-median accuracy {v['accuracy']:.1%}")  # fmt: skip
    print("\n=== Ranking (mean of the two ranks; lower is better) ===")
    reg = Registry.load()
    for m, sc in overall.items():
        ur = rets[m]["ic"] > rets["ridge (baseline)"]["ic"] and rets[m]["t_vs_ridge"] >= 3.0
        uv = vols[m]["qlike"] < vols["HAR (baseline)"]["qlike"] and vols[m]["dm_t_vs_har"] >= 3.0
        print(f"  {m:18s} rank R {int(rank_r[m])}, rank V {int(rank_v[m])}, mean {sc:.1f} | "
              f"useful for returns: {'YES' if ur else 'no'}; for volatility: "
              f"{'YES' if uv else 'no'}")  # fmt: skip
        if m == "arima_garch":  # the only model whose forecasts changed: log a corrected trial
            metrics = {"return_ic": rets[m]["ic"], "return_ic_t": rets[m]["ic_t"],
                       "vol_qlike": vols[m]["qlike"], "vol_dm_t_vs_har": vols[m]["dm_t_vs_har"],
                       "vol_accuracy": vols[m]["accuracy"], "overall_rank": float(sc),
                       "n": len(fixed)}  # fmt: skip
            t = log_external_trial(
                "shootout_arima_garch_fixed",
                "promoted" if (ur or uv) else "rejected_holdout",
                "ARIMA-GARCH (fit only on real history) beats the baselines",
                "forecaster shootout v1, after the zero-padding bug fix",
                metrics,
                True,
                __file__,
                notes=f"pre-registered in {PREREG}; rerun after a bug found post-result",
            )
            print(f"    -> trial #{t.trial_id} (corrected)")
            if uv:
                reg.record(SignalRecord("shootout_arima_garch_vol", "ARIMA-GARCH 20-day vol",
                                        "risk", "qlike_dm_t_vs_har", vols[m]["qlike"],
                                        vols[m]["dm_t_vs_har"], "frozen 2024-2026",
                                        f"trial #{t.trial_id}"))  # fmt: skip
    reg.save()
    print(f"\nBEST OVERALL: {overall.index[0]}")


if __name__ == "__main__":
    main()
