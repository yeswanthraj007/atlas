"""Forecaster shootout v1, exactly as pre-registered in
research/preregistered/2026-11-07-forecaster-shootout-v1.md (including Amendment 1).

    HF_HUB_OFFLINE=1 uv run python scripts/forecaster_shootout_v1.py            # full run
    HF_HUB_OFFLINE=1 uv run python scripts/forecaster_shootout_v1.py --smoke    # plumbing check

--smoke uses 60 stocks, tiny training budgets and scores on 2023 (never 2024+), logs no trials.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Ridge

sys.path.insert(0, str(Path(__file__).parent))
from market_transformer_v2 import CHRONOS, CHRONOS_REVISION, EMB_DIM, news_inputs

from atlas.lab.trials import log_external_trial
from atlas.ml.forecasters import (
    H,
    LSTMBackbone,
    LSTMVol,
    VolConfig,
    arma_garch_forecast,
    fit_arma_garch,
    fit_vol,
    predict_vol,
    score_return_task,
    score_vol_task,
    t_stat,
)
from atlas.ml.market_gpt import (
    MarketGPT,
    RankConfig,
    ScoreModel,
    build_panel,
    fit_ranker,
    predict_rows,
    predict_scores,
    sample_index,
)
from atlas.ml.text_embed import device
from atlas.ml.volatility import log_realized_vol
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

PREREG = "research/preregistered/2026-11-07-forecaster-shootout-v1.md"
TIMESFM = "google/timesfm-2.5-200m-pytorch"
TIMESFM_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
CONTEXT = 252
V3 = Path("data/reports/market_gpt_v3")
V2 = Path("data/reports/market_transformer_v2")
_MODELS: dict = {}  # pretrained models, loaded once
MODELS = ["arima_garch", "lstm", "marketgpt", "chronos", "timesfm"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def _arma_one(r_pct: np.ndarray, cut: int, positions: np.ndarray):
    """Fit on returns up to `cut`, forecast at `positions`; (ret, vol) or None."""
    import warnings

    warnings.simplefilter("ignore")
    m = fit_arma_garch(r_pct[: cut + 1])
    return None if m is None else arma_garch_forecast(m, r_pct, positions)


def train_split(rows: pd.DataFrame, end: pd.Timestamp, val_year: int):
    ok = rows["fwd_ret"].notna() & rows["fwd_log_vol"].notna()
    tr = rows[ok & (rows["label_end"] <= end)]
    va = rows[ok & (rows["date"].dt.year == val_year) & (rows["label_end"].dt.year == val_year)]
    return tr, va


def chronos_forecast(ctx: np.ndarray, progress=log) -> np.ndarray:
    """Median forecasts (n, H) for contexts (n, CONTEXT)."""
    from chronos import BaseChronosPipeline

    if "chronos" not in _MODELS:
        _MODELS["chronos"] = BaseChronosPipeline.from_pretrained(
            CHRONOS, revision=CHRONOS_REVISION, device_map=str(device()), torch_dtype=torch.float32
        )
    pipe = _MODELS["chronos"]
    out = []
    for i in range(0, len(ctx), 512):
        q, _ = pipe.predict_quantiles(
            torch.tensor(ctx[i : i + 512], dtype=torch.float32),
            prediction_length=H,
            quantile_levels=[0.5],
        )
        out.append(q[:, :, 0].numpy())
    return np.concatenate(out)


def timesfm_forecast(ctx: np.ndarray, progress=log) -> np.ndarray:
    """Point forecasts (n, H) for contexts (n, CONTEXT) from TimesFM 2.5 (zero-shot)."""
    import timesfm

    if "timesfm" not in _MODELS:
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM, revision=TIMESFM_REVISION)
        m.compile(
            timesfm.ForecastConfig(
                max_context=CONTEXT,
                max_horizon=128,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )
        _MODELS["timesfm"] = m
    model = _MODELS["timesfm"]
    out = []
    for i in range(0, len(ctx), 256):
        point, _ = model.forecast(horizon=H, inputs=list(ctx[i : i + 256].astype(np.float32)))
        out.append(np.asarray(point)[:, :H])
        if (i // 256) % 10 == 0:
            progress(f"  timesfm {min(i + 256, len(ctx)):,}/{len(ctx):,}")
    return np.concatenate(out)


def main(smoke: bool) -> None:
    out_dir = Path("data/reports/shootout_v1" + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"pre-registration: {PREREG}{'  [SMOKE: scored on 2023, nothing logged]' if smoke else ''}")
    rd = load_research_data("price", universe="sp500")
    idx = rd.closes.index
    if smoke:
        keep = list(rd.bars)[::8][:60]
        rd.bars = {k: rd.bars[k] for k in keep}
        rd.closes = rd.closes[keep]
    closes = rd.closes
    cols = list(closes.columns)

    test_start = pd.Timestamp("2023-01-03" if smoke else "2024-01-02", tz="UTC")
    p0 = int(idx.searchsorted(test_start))
    last = int(idx.searchsorted(pd.Timestamp("2023-12-29" if smoke else "2100-01-01", tz="UTC")))
    positions = np.arange(p0, min(last, len(idx) - H - 1), H)
    log(f"{len(positions)} test dates, {idx[positions[0]].date()} .. {idx[positions[-1]].date()}")
    cut = p0 - 1  # parameters and training use data strictly before the first test date

    if smoke:
        rng = np.random.default_rng(0)
        pos = np.arange(0, len(idx), 7)
        news = {k: (pos, np.ones(len(pos)), rng.normal(size=(len(pos), EMB_DIM)).astype(np.float16))
                for k in cols}  # fmt: skip
    else:
        news = news_inputs(rd, idx, progress=log)
    panel = build_panel(rd.bars, rd.benchmark, idx, news, EMB_DIM)
    rows = sample_index(panel, stride=1, membership=rd.membership)
    hist = closes.notna().astype(int).rolling(CONTEXT + 21, min_periods=CONTEXT + 21).sum()
    hist_ok = (hist == CONTEXT + 21).to_numpy()
    te = rows[rows["end"].isin(positions) & rows["fwd_ret"].notna() & rows["fwd_log_vol"].notna()]
    te = te[hist_ok[te["end"].to_numpy(), te["stock"].to_numpy()]].reset_index(drop=True)
    log(f"{len(te):,} candidate (date, stock) pairs in the test set")
    p, j = te["end"].to_numpy(), te["stock"].to_numpy()
    val_year = idx[cut].year  # the last full year before the first test date
    tr_rows, va_rows = train_split(rows, pd.Timestamp(f"{val_year - 1}-12-31", tz="UTC"), val_year)
    log(f"LSTM data: train {len(tr_rows):,} rows | validation {len(va_rows):,} (year {val_year})")

    pred_ret: dict[str, np.ndarray] = {}
    pred_vol: dict[str, np.ndarray] = {}  # log annualised volatility

    # ---- 1. ARIMA-GARCH ----
    t0 = time.time()
    logret_pct = (100 * np.log(closes / closes.shift(1))).to_numpy()
    by_stock = {s: np.where(j == s)[0] for s in np.unique(j)}
    jobs = []
    for s, ix in by_stock.items():
        r = np.nan_to_num(logret_pct[:, s], nan=0.0)
        jobs.append((s, ix, delayed(_arma_one)(r, cut, p[ix])))
    res = Parallel(n_jobs=8)(job for *_, job in jobs)
    ag_ret, ag_vol = np.full(len(te), np.nan), np.full(len(te), np.nan)
    for (_, ix, _), r in zip(jobs, res, strict=True):
        if r is not None:
            ag_ret[ix], ag_vol[ix] = r[0], np.log(r[1])
    pred_ret["arima_garch"], pred_vol["arima_garch"] = ag_ret, ag_vol
    log(
        f"ARIMA-GARCH: fitted {int(np.isfinite(ag_ret).sum()):,}/{len(te):,} "
        f"in {time.time() - t0:.0f}s"
    )

    # ---- 2. LSTM ----
    smoke_cfg = dict(samples_per_epoch=20_000, epochs=2) if smoke else {}
    path_r, path_v = out_dir / "lstm_return.pt", out_dir / "lstm_vol.pt"
    lstm_r = ScoreModel(LSTMBackbone())
    if path_r.exists():
        lstm_r.load_state_dict(torch.load(path_r, map_location="cpu"))
        lstm_r = lstm_r.to(device()).eval()
    else:
        log("training LSTM (returns, rank loss)")
        cfg = RankConfig(loss="rank", lr=1e-3, drop_market=True, val_every_nth_date=5, **smoke_cfg)
        lstm_r = fit_ranker(lstm_r, panel, tr_rows, va_rows, cfg, False, progress=log)
        torch.save(lstm_r.state_dict(), path_r)
    pred_ret["lstm"] = predict_scores(lstm_r, panel, te, False, drop_market=True)
    lstm_v = LSTMVol(LSTMBackbone())
    if path_v.exists():
        lstm_v.load_state_dict(torch.load(path_v, map_location="cpu"))
        lstm_v = lstm_v.to(device()).eval()
    else:
        log("training LSTM (volatility, MSE)")
        lstm_v = fit_vol(lstm_v, panel, tr_rows, va_rows, VolConfig(**smoke_cfg), progress=log)
        torch.save(lstm_v.state_dict(), path_v)
    pred_vol["lstm"] = predict_vol(lstm_v, panel, te)

    # ---- 3. MarketGPT (frozen models from v3 and v2) ----
    mg = np.zeros(len(te))
    year = te["date"].dt.year.to_numpy()
    for refit, ys in (("A", (2023, 2024)), ("B", (2025,)), ("C", (2026,))):
        m = ScoreModel(MarketGPT(EMB_DIM, width=256))
        m.load_state_dict(torch.load(V3 / f"price_{refit}_s0.pt", map_location="cpu"))
        sel = np.isin(year, ys)
        if sel.any():
            mg[sel] = predict_scores(m.to(device()).eval(), panel, te[sel], False, drop_market=True)
    pred_ret["marketgpt"] = mg
    vt = MarketGPT(EMB_DIM)
    vt.load_state_dict(torch.load(V2 / "vol_transformer.pt", map_location="cpu"))
    pred_vol["marketgpt"] = predict_rows(vt.to(device()).eval(), "vol", panel, te, True)

    # ---- 4/5. Chronos and TimesFM (zero-shot) ----
    logret = np.log(closes / closes.shift(1)).to_numpy(np.float32)
    lrv20 = log_realized_vol(closes, 20).to_numpy(np.float32)
    off = np.arange(-CONTEXT + 1, 1)
    ctx_ret = np.stack([logret[p[i] + off, j[i]] for i in range(len(te))])
    ctx_vol = np.stack([lrv20[p[i] + off, j[i]] for i in range(len(te))])
    good = np.isfinite(ctx_ret).all(1) & np.isfinite(ctx_vol).all(1)
    log(f"foundation-model contexts complete for {int(good.sum()):,}/{len(te):,}")
    for name, fn in (("chronos", chronos_forecast), ("timesfm", timesfm_forecast)):
        t0 = time.time()
        r_out, v_out = np.full(len(te), np.nan), np.full(len(te), np.nan)
        fr = fn(ctx_ret[good])
        fv = fn(ctx_vol[good])
        r_out[good], v_out[good] = fr.sum(axis=1), fv[:, H - 1]
        pred_ret[name], pred_vol[name] = r_out, v_out
        log(f"{name}: done in {time.time() - t0:.0f}s")

    # ---- baselines ----
    feats = {w: log_realized_vol(closes, w).to_numpy(np.float32) for w in (5, 21, 63, 252)}
    c = closes
    rank = lambda f: f.rank(axis=1, pct=True).to_numpy(np.float32)  # noqa: E731
    r_feats = [rank(-(c / c.shift(21) - 1)), rank(c.shift(21) / c.shift(252) - 1), rank(
        pd.DataFrame(feats[252], index=c.index, columns=c.columns))]  # fmt: skip
    trn = tr_rows[tr_rows["end"] % H == 0]
    x_tr = np.column_stack([f[trn["end"], trn["stock"]] for f in r_feats])
    y_tr = trn.groupby("date")["fwd_ret"].rank(pct=True).to_numpy()
    okr = np.isfinite(x_tr).all(1)
    ridge = Ridge(alpha=10.0).fit(x_tr[okr], y_tr[okr])
    x_te = np.column_stack([f[p, j] for f in r_feats])
    base_ret = ridge.predict(np.nan_to_num(x_te, nan=0.5))
    har_cols = [5, 21, 63]
    h_tr = np.column_stack([feats[w][tr_rows["end"], tr_rows["stock"]] for w in har_cols])
    okh = np.isfinite(h_tr).all(1) & tr_rows["fwd_log_vol"].notna().to_numpy()
    har = LinearRegression().fit(h_tr[okh], tr_rows["fwd_log_vol"].to_numpy()[okh])
    h_te = np.column_stack([feats[w][p, j] for w in har_cols])
    base_har = har.predict(np.nan_to_num(h_te, nan=np.nanmean(h_tr)))
    base_naive = feats[21][p, j]

    # ---- evaluation set: every model has a forecast ----
    allp = [pred_ret[m] for m in MODELS] + [pred_vol[m] for m in MODELS]
    allp += [base_ret, base_har, base_naive]
    keep = np.isfinite(np.column_stack(allp)).all(1)
    ev = te[keep].reset_index(drop=True)
    log(f"EVALUATION SET: {len(ev):,} (date, stock) pairs over {ev['date'].nunique()} dates")
    har_ev = base_har[keep]
    saved = ev[["date", "instrument", "fwd_ret", "fwd_log_vol"]].copy()
    for m in MODELS:
        saved[f"ret_{m}"], saved[f"vol_{m}"] = pred_ret[m][keep], pred_vol[m][keep]
    saved["ret_ridge"], saved["vol_har"], saved["vol_naive"] = (
        base_ret[keep],
        har_ev,
        base_naive[keep],
    )
    saved.to_parquet(out_dir / "predictions.parquet")  # the ensemble stage reads this

    rets = {m: score_return_task(ev, pred_ret[m][keep]) for m in MODELS}
    rets["ridge (baseline)"] = score_return_task(ev, base_ret[keep])
    vols = {m: score_vol_task(ev, pred_vol[m][keep], har_ev) for m in MODELS}
    vols["HAR (baseline)"] = score_vol_task(ev, har_ev, har_ev)
    vols["naive (baseline)"] = score_vol_task(ev, base_naive[keep], har_ev)

    ridge_ic = rets["ridge (baseline)"]["ic_by_date"]
    for m in MODELS:
        rets[m]["t_vs_ridge"] = t_stat(rets[m]["ic_by_date"] - ridge_ic)
    rank_r = pd.Series({m: rets[m]["ic"] for m in MODELS}).rank(ascending=False)
    rank_v = pd.Series({m: vols[m]["qlike"] for m in MODELS}).rank(ascending=True)
    overall = ((rank_r + rank_v) / 2).sort_values()

    print("\n=== Task R: 20-day return ranking (IC; baseline ridge IC "
          f"{rets['ridge (baseline)']['ic']:+.4f}) ===")  # fmt: skip
    for m in [*MODELS, "ridge (baseline)"]:
        r = rets[m]
        extra = f" | t vs ridge {r['t_vs_ridge']:+.2f}" if m in MODELS else ""
        print(
            f"  {m:18s} IC {r['ic']:+.4f} (t {r['ic_t']:+.2f}) "
            f"| accuracy {r['accuracy']:.1%}{extra}"
        )
    print("\n=== Task V: 20-day volatility (QLIKE, lower is better) ===")
    for m in [*MODELS, "HAR (baseline)", "naive (baseline)"]:
        v = vols[m]
        print(f"  {m:18s} QLIKE {v['qlike']:.4f} | DM t vs HAR {v['dm_t_vs_har']:+.2f} | "
              f"above-median accuracy {v['accuracy']:.1%}")  # fmt: skip
    print("\n=== Ranking (mean of the two ranks; lower is better) ===")
    for m, s in overall.items():
        ur = rets[m]["ic"] > rets["ridge (baseline)"]["ic"] and rets[m]["t_vs_ridge"] >= 3.0
        uv = vols[m]["qlike"] < vols["HAR (baseline)"]["qlike"] and vols[m]["dm_t_vs_har"] >= 3.0
        print(
            f"  {m:18s} rank R {int(rank_r[m])}, rank V {int(rank_v[m])}, mean {s:.1f} | "
            f"useful for returns: {'YES' if ur else 'no'}; "
            f"for volatility: {'YES' if uv else 'no'}"
        )
    print(f"\nBEST OVERALL: {overall.index[0]}")
    if smoke:
        return

    reg = Registry.load()
    for m in MODELS:
        ur = rets[m]["ic"] > rets["ridge (baseline)"]["ic"] and rets[m]["t_vs_ridge"] >= 3.0
        uv = vols[m]["qlike"] < vols["HAR (baseline)"]["qlike"] and vols[m]["dm_t_vs_har"] >= 3.0
        metrics = {
            "return_ic": rets[m]["ic"], "return_ic_t": rets[m]["ic_t"],
            "return_accuracy": rets[m]["accuracy"], "return_t_vs_ridge": rets[m]["t_vs_ridge"],
            "vol_qlike": vols[m]["qlike"], "vol_dm_t_vs_har": vols[m]["dm_t_vs_har"],
            "vol_accuracy": vols[m]["accuracy"], "rank_return": int(rank_r[m]),
            "rank_vol": int(rank_v[m]), "overall_rank": float(overall[m]),
            "n": len(ev), "dates": int(ev["date"].nunique()),
        }  # fmt: skip
        t = log_external_trial(
            f"shootout_{m}", "promoted" if (ur or uv) else "rejected_holdout",
            f"{m} forecasts 20-day returns / volatility better than the baselines",
            "forecaster shootout v1", metrics, True, __file__, notes=f"pre-registered in {PREREG}",
        )  # fmt: skip
        print(f"  trial #{t.trial_id}: {m}: {'PROMOTED' if (ur or uv) else 'rejected'}")
        if uv:
            reg.record(SignalRecord(f"shootout_{m}_vol", f"{m} 20-day volatility", "risk",
                                    "qlike_dm_t_vs_har", vols[m]["qlike"], vols[m]["dm_t_vs_har"],
                                    "frozen 2024-2026", f"trial #{t.trial_id}"))  # fmt: skip
        if ur:
            reg.record(SignalRecord(f"shootout_{m}_ret", f"{m} 20-day return ranking", "direction",
                                    "ic_20d", rets[m]["ic"], rets[m]["ic_t"],
                                    "frozen 2024-2026", f"trial #{t.trial_id}"))  # fmt: skip
    reg.save()
    (out_dir / "results.json").write_text(json.dumps({
        "returns": {k: {a: b for a, b in v.items() if a != "ic_by_date"} for k, v in rets.items()},
        "volatility": vols, "ranking": overall.to_dict(), "n": len(ev),
    }, indent=2, default=float))  # fmt: skip
    log("done")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
