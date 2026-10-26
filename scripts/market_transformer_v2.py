"""Market transformer v2 (price + news, GPT-style pretraining) and time-series forecasting,
exactly as pre-registered in research/preregistered/2026-10-26-market-transformer-v2.md.

    uv run python scripts/market_transformer_v2.py
"""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

from atlas.backtest.periodic import decile_aims, simulate
from atlas.lab.trials import log_external_trial
from atlas.marketdata.news import load_news
from atlas.ml.market_gpt import (
    GPTConfig,
    MarketGPT,
    build_panel,
    fit_stage,
    predict_rows,
    qlike,
    sample_index,
)
from atlas.ml.news_model import explode_by_symbol, identity_tokens, news_day_index
from atlas.ml.text_embed import device, embed_cached
from atlas.ml.volatility import log_realized_vol
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

PREREG = "research/preregistered/2026-10-26-market-transformer-v2.md"
OUT = Path("data/reports/market_transformer_v2")
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
VAL_YEAR = 2023
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
EMB_DIM = 64
CHRONOS = "amazon/chronos-bolt-small"
CHRONOS_REVISION = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "COST", "JPM", "XOM", "INTC", "META"]


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def monthly(rows: pd.DataFrame) -> pd.DataFrame:
    """Every 4th sample date: non-overlapping 20-day outcomes."""
    dates = sorted(rows["date"].unique())[::4]
    return rows[rows["date"].isin(dates)]


def eval_returns(name: str, m: pd.DataFrame, score: np.ndarray, cost: pd.DataFrame) -> dict:
    d = m.assign(p=score)
    ic = d.groupby("date").apply(lambda g: stats.spearmanr(g["p"], g["fwd_ret"]).statistic)
    med = d.groupby("date")["fwd_ret"].transform("median")
    centre = d.groupby("date")["p"].transform("median")
    pred = d.pivot(index="date", columns="instrument", values="p")
    fwd = d.pivot(index="date", columns="instrument", values="fwd_ret").fillna(0.0)
    c = cost.reindex(index=pred.index, columns=pred.columns)
    ls = {tau: simulate(decile_aims(pred), fwd, c, tau) for tau in (1.0, 0.5)}
    per = {}
    for tk in TICKERS:
        x = d[d["instrument"] == f"EQ:{tk}"]
        if len(x):
            right = (x["p"] > centre[x.index]) == (x["fwd_ret"] > med[x.index])
            per[tk] = float(right.mean())
    return {
        "model": name,
        "ic": float(ic.mean()),
        "ic_t": tstat(ic),
        "months": len(ic),
        "accuracy_vs_median": float(((d["p"] > centre) == (d["fwd_ret"] > med)).mean()),
        "ls_gross": float(ls[1.0].gross.mean()),
        "ls_net_tau1": float(ls[1.0].net.mean()),
        "ls_net_tau05": float(ls[0.5].net.mean()),
        "per_ticker": per,
        "_ic_series": ic,
    }


def eval_vol(name: str, m: pd.DataFrame, pred: np.ndarray, har: np.ndarray) -> dict:
    d = m.assign(p=pred, h=har)
    d["loss"] = qlike(d["p"].to_numpy(), d["fwd_log_vol"].to_numpy())
    d["loss_har"] = qlike(d["h"].to_numpy(), d["fwd_log_vol"].to_numpy())
    diff = d.groupby("date").apply(lambda g: (g["loss_har"] - g["loss"]).mean())
    above = d.groupby("date").apply(
        lambda g: (
            (g["p"] > g["p"].median()) == (g["fwd_log_vol"] > g["fwd_log_vol"].median())
        ).mean()
    )
    return {
        "model": name,
        "qlike": float(d["loss"].mean()),
        "qlike_har": float(d["loss_har"].mean()),
        "dm_t_vs_har": tstat(diff),  # > 0: lower loss than HAR
        "accuracy_above_median": float(above.mean()),
        "months": len(diff),
    }


def news_inputs(rd, dates: pd.DatetimeIndex, progress=print):
    """Daily mean of reduced, identity-masked news embeddings per stock."""
    syms = [c.split(":")[1] for c in rd.closes.columns]
    news = load_news()
    rows = explode_by_symbol(news, set(syms))
    rows["d0"] = news_day_index(rows["created_at"], dates)
    rows = rows[rows["d0"] >= 0].reset_index(drop=True)
    train_pos = dates.searchsorted(TRAIN_END, side="right") - 1
    train = rows[rows["d0"] <= train_pos]
    masked = identity_tokens(train)
    mask_id = hashlib.sha1(" ".join(sorted(masked)).encode()).hexdigest()[:10]
    progress(
        f"news: {len(news):,} articles, {len(rows):,} article-stock pairs, "
        f"{len(masked):,} identity words masked (learned from news up to 2022)"
    )
    emb = embed_cached(rows, masked, f"data/news_emb/embeddings_{mask_id}.npz", progress=progress)
    pca = PCA(EMB_DIM, random_state=0).fit(emb[(rows["d0"] <= train_pos).to_numpy()])
    progress(
        f"PCA to {EMB_DIM} dims (fit on 2016-2022 news) keeps "
        f"{pca.explained_variance_ratio_.sum():.0%} of the variance"
    )
    z = pca.transform(emb).astype(np.float32)
    out = {}
    df = pd.DataFrame(z).assign(sym=rows["symbol"].to_numpy(), d0=rows["d0"].to_numpy())
    for sym, g in df.groupby("sym"):
        day = g.groupby("d0")
        mean = day[list(range(EMB_DIM))].mean()
        out[f"EQ:{sym}"] = (
            mean.index.to_numpy(),
            day.size().to_numpy(),
            mean.to_numpy().astype(np.float16),
        )
    return out


def chronos_forecasts(closes: pd.DataFrame, rows: pd.DataFrame, progress=print):
    """Zero-shot Chronos-Bolt: 20-day return and log-volatility forecasts per row."""
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(
        CHRONOS, revision=CHRONOS_REVISION, device_map=str(device()), torch_dtype=torch.float32
    )
    logret = np.log(closes / closes.shift(1)).to_numpy()
    col = {c: i for i, c in enumerate(closes.columns)}
    pos = closes.index.get_indexer(rows["date"])
    j = rows["instrument"].map(col).to_numpy()
    ret_f, vol_f = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for start in range(0, len(rows), 512):
        sl = slice(start, start + 512)
        ctx = np.stack([logret[p - 251 : p + 1, k] for p, k in zip(pos[sl], j[sl], strict=True)])
        r_ctx = torch.tensor(ctx, dtype=torch.float32)
        q, _ = pipe.predict_quantiles(r_ctx, prediction_length=20, quantile_levels=[0.5])
        ret_f[sl] = np.expm1(q[:, :, 0].sum(1).numpy())
        _, mean_abs = pipe.predict_quantiles(
            r_ctx.abs(), prediction_length=20, quantile_levels=[0.5]
        )
        sd = mean_abs.clamp(min=1e-6).mean(1).numpy() * np.sqrt(np.pi / 2)  # E|r| -> sigma
        vol_f[sl] = np.log(sd * np.sqrt(252))
        progress(f"  chronos {min(start + 512, len(rows)):,}/{len(rows):,}")
    return ret_f, vol_f


def _load(model: MarketGPT, path: Path) -> MarketGPT:
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device()).eval()


def main(smoke: bool = False, resume: bool = False) -> None:
    """smoke=True: 25 stocks, random news vectors, 1 epoch, nothing logged (pipeline check).
    resume=True: load the frozen models saved by an interrupted run instead of training."""
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pre-registration: {PREREG}")
    t0 = time.time()
    rd = load_research_data("price", universe="sp500")
    dates = rd.closes.index
    if smoke:
        keep = list(rd.bars)[:25]
        rd.bars = {k: rd.bars[k] for k in keep}
        rd.closes = rd.closes[keep]
        rng = np.random.default_rng(0)
        pos = np.arange(0, len(dates), 7)
        news = {
            k: (pos, np.ones(len(pos)), rng.normal(size=(len(pos), EMB_DIM)).astype(np.float16))
            for k in keep
        }
    else:
        news = news_inputs(rd, dates)
    panel = build_panel(rd.bars, rd.benchmark, dates, news, EMB_DIM)
    rows = sample_index(panel, stride=5, membership=rd.membership)
    print(
        f"{len(rows):,} samples, {len(panel.instruments)} stocks, "
        f"news on {(panel.news_meta[:, :, 1] == 0).mean():.1%} of stock-days "
        f"({(time.time() - t0) / 60:.0f} min)"
    )

    train_pos = dates.searchsorted(TRAIN_END, side="right") - 1
    val_end = dates.searchsorted(pd.Timestamp(f"{VAL_YEAR}-12-31", tz="UTC"), side="right") - 1
    yr = rows["date"].dt.year
    pre_tr = rows[rows["end"] + 1 <= train_pos]
    pre_va = rows[(yr == VAL_YEAR) & (rows["end"] + 1 <= val_end)]
    known = rows["target"].notna()
    ft_tr = rows[known & (rows["label_end"] <= TRAIN_END)]
    ft_va = rows[known & (yr == VAL_YEAR) & (rows["label_end"].dt.year == VAL_YEAR)]
    test = rows[(rows["date"] >= TEST_START) & rows["fwd_ret"].notna()]
    m_test = monthly(test).reset_index(drop=True)
    print(
        f"pretrain {len(pre_tr):,} / {len(pre_va):,} | fine-tune {len(ft_tr):,} / "
        f"{len(ft_va):,} | test {len(test):,} ({len(m_test):,} in monthly evaluation)"
    )

    ret_results, vol_results, ret_scores = [], [], {}
    backbones = {}
    for name, use_news in (("market_gpt_news", True), ("market_gpt_price", False)):
        print(f"\n== {name} ==")
        model = MarketGPT(EMB_DIM)
        print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
        if resume:
            model = _load(model, OUT / f"{name}.pt")
        else:
            cfg = GPTConfig(pretrain_epochs=1, finetune_epochs=1) if smoke else GPTConfig()
            model = fit_stage(model, "pretrain", panel, pre_tr, pre_va, cfg, use_news)
            backbones[name] = copy.deepcopy(model)
            model = fit_stage(model, "classify", panel, ft_tr, ft_va, cfg, use_news)
            torch.save(model.state_dict(), OUT / f"{name}.pt")
        ret_scores[name] = predict_rows(model, "classify", panel, m_test, use_news)
        ret_results.append(eval_returns(name, m_test, ret_scores[name], rd.spread))

    # D: volatility transformer on the pretrained news backbone
    print("\n== vol_transformer ==")
    vtr, vva = ft_tr[ft_tr["fwd_log_vol"].notna()], ft_va[ft_va["fwd_log_vol"].notna()]
    if resume:
        vol_model = _load(MarketGPT(EMB_DIM), OUT / "vol_transformer.pt")
    else:
        vol_model = backbones["market_gpt_news"]
        vol_cfg = GPTConfig(finetune_epochs=1) if smoke else GPTConfig()
        vol_model = fit_stage(vol_model, "vol", panel, vtr, vva, vol_cfg, True)
        torch.save(vol_model.state_dict(), OUT / "vol_transformer.pt")
    mv = m_test[m_test["fwd_log_vol"].notna()].reset_index(drop=True)
    vol_pred = predict_rows(vol_model, "vol", panel, mv, True)

    # HAR baseline (Corsi 2009) on the same rows, fit on the training period
    har_feats = {w: log_realized_vol(rd.closes, w) for w in (5, 21, 63)}

    def har_x(r: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([r["date"], r["instrument"]])
        cols = [f.stack(future_stack=True).reindex(key).to_numpy() for f in har_feats.values()]
        return np.column_stack(cols)

    xh = har_x(vtr)
    ok = np.isfinite(xh).all(1)
    har = LinearRegression().fit(xh[ok], vtr["fwd_log_vol"].to_numpy()[ok])
    xm = har_x(mv)
    har_pred = har.predict(np.nan_to_num(xm, nan=np.nanmean(xm)))

    # C: Chronos-Bolt zero-shot
    print("\n== chronos_bolt (zero-shot) ==")
    c_ret, c_vol = chronos_forecasts(rd.closes, m_test)
    ret_results.append(eval_returns("chronos_returns", m_test, c_ret, rd.spread))
    cv = pd.Series(c_vol, index=m_test.index)[m_test["fwd_log_vol"].notna()].to_numpy()
    vol_results.append(eval_vol("chronos_vol", mv, cv, har_pred))
    vol_results.append(eval_vol("vol_transformer", mv, vol_pred, har_pred))

    # ridge baseline, v1 specification, on these samples
    c = rd.closes
    feats = [
        -(c / c.shift(21) - 1),
        c.shift(21) / c.shift(252) - 1,
        c.pct_change().rolling(252).std(),
    ]

    def ridge_x(r: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([r["date"], r["instrument"]])
        return np.column_stack(
            [
                f.rank(axis=1, pct=True)
                .stack(future_stack=True)
                .reindex(key)
                .fillna(0.5)
                .to_numpy()
                - 0.5
                for f in feats
            ]
        )

    ridge = Ridge(alpha=10.0).fit(ridge_x(ft_tr), ft_tr["target"].to_numpy())
    base = eval_returns("ridge_baseline", m_test, ridge.predict(ridge_x(m_test)), rd.spread)

    # report
    print("\n=== Frozen test, 2024-2026, S&P 500 (monthly, non-overlapping) ===")
    print(f"ridge baseline: IC {base['ic']:+.4f} (t {base['ic_t']:+.2f})")
    diff = (ret_results[0]["_ic_series"] - ret_results[1]["_ic_series"]).dropna()
    news_t = tstat(diff)
    print(
        f"news adds value? IC(news) - IC(price only) = {diff.mean():+.4f}, paired t "
        f"{news_t:+.2f} -> {'YES' if news_t >= 2 else 'no'}"
    )
    if smoke:
        for r in [*ret_results, *vol_results]:
            print({k: v for k, v in r.items() if not k.startswith("_")})
        return
    reg = Registry.load()
    for r in ret_results:
        print(
            f"\n{r['model']}: IC {r['ic']:+.4f} (t {r['ic_t']:+.2f}, {r['months']} months) | "
            f"beats-median accuracy {r['accuracy_vs_median']:.1%}"
        )
        print(
            f"  top-minus-bottom decile per month: gross {r['ls_gross']:+.2%}, net (tau 1) "
            f"{r['ls_net_tau1']:+.2%}, net (tau 0.5) {r['ls_net_tau05']:+.2%}"
        )
        print(
            "  per ticker (right vs median): "
            + ", ".join(f"{k} {v:.0%}" for k, v in r["per_ticker"].items())
        )
        checks = {
            "IC t >= 2": r["ic"] > 0 and r["ic_t"] >= 2.0,
            "net decile spread > 0 (tau 0.5)": r["ls_net_tau05"] > 0,
            "beats ridge baseline IC": r["ic"] > base["ic"],
        }
        passed = all(checks.values())
        for k, ok_ in checks.items():
            print(f"  {'PASS' if ok_ else 'FAIL'}  {k}")
        metrics = {k: v for k, v in r.items() if not k.startswith("_")}
        metrics["ridge_baseline_ic"] = base["ic"]
        if r["model"] == "market_gpt_news":
            metrics["news_vs_price_paired_t"] = news_t
        trial = log_external_trial(
            r["model"],
            "promoted" if passed else "rejected_holdout",
            f"{r['model']} predicts which S&P 500 stocks beat the median over 20 days",
            "GPT-style pretraining on price+news sequences; Chronos-Bolt zero-shot",
            metrics,
            True,
            __file__,
            notes=f"pre-registered in {PREREG}",
        )
        print(f"  -> trial #{trial.trial_id}: {'PROMOTED' if passed else 'rejected'}")
        if passed:
            reg.record(
                SignalRecord(
                    r["model"],
                    f"{r['model']} 20-day direction",
                    "direction",
                    "ic_20d",
                    r["ic"],
                    r["ic_t"],
                    "frozen 2024-2026",
                    f"trial #{trial.trial_id}",
                )
            )
    for r in vol_results:
        passed = r["qlike"] < r["qlike_har"] and r["dm_t_vs_har"] >= 2.0
        print(
            f"\n{r['model']}: QLIKE {r['qlike']:.4f} vs HAR {r['qlike_har']:.4f} "
            f"(Diebold-Mariano t {r['dm_t_vs_har']:+.2f}) | above-median accuracy "
            f"{r['accuracy_above_median']:.1%}\n  {'PASS' if passed else 'FAIL'}  beats HAR"
        )
        trial = log_external_trial(
            r["model"],
            "promoted" if passed else "rejected_holdout",
            f"{r['model']} forecasts next-20-day volatility better than HAR",
            "transformer / pretrained time-series model vs Corsi (2009) HAR",
            r,
            True,
            __file__,
            notes=f"pre-registered in {PREREG}",
        )
        print(f"  -> trial #{trial.trial_id}: {'PROMOTED' if passed else 'rejected'}")
        if passed:
            reg.record(
                SignalRecord(
                    r["model"],
                    f"{r['model']} 20-day volatility",
                    "risk",
                    "qlike_dm_t_vs_har",
                    r["qlike"],
                    r["dm_t_vs_har"],
                    "frozen 2024-2026",
                    f"trial #{trial.trial_id}",
                )
            )
    reg.save()
    pd.DataFrame(
        [
            {k: v for k, v in r.items() if not isinstance(v, (dict, pd.Series))}
            for r in [*ret_results, base, *vol_results]
        ]
    ).to_csv(OUT / "results.csv", index=False)
    print(f"\ndone in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    import sys

    main(smoke="--smoke" in sys.argv, resume="--resume" in sys.argv)
