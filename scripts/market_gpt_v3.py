"""MarketGPT v3, exactly as pre-registered in research/preregistered/2026-10-26-market-gpt-v3.md.

    HF_HUB_OFFLINE=1 uv run python scripts/market_gpt_v3.py            # full run
    HF_HUB_OFFLINE=1 uv run python scripts/market_gpt_v3.py --smoke    # pipeline check

Every trained model is saved as soon as it is trained and reused if the run restarts (all
models are frozen once saved). --smoke uses 60 stocks, random news and tiny budgets. It computes no
final test metrics, but refits B and C validate on 2024 and 2025 by design, so their
(meaningless, tiny-budget) validation ICs are printed.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from market_transformer_v2 import EMB_DIM, eval_returns, news_inputs, tstat

from atlas.lab.trials import log_external_trial
from atlas.ml.market_gpt import (
    GPTConfig,
    MarketGPT,
    RankConfig,
    ScoreModel,
    build_panel,
    fit_ranker,
    fit_stage,
    predict_scores,
    sample_index,
)
from atlas.ml.text_embed import device
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

PREREG = "research/preregistered/2026-10-26-market-gpt-v3.md"
PRETRAIN_END = 2022
REFITS = {
    "A": 2022,
    "B": 2023,
    "C": 2024,
}  # trains on labels known by year-end; val = +1; predicts +2
SEEDS = (0, 1, 2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def split(rows: pd.DataFrame, train_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = rows["target"].notna()
    end = pd.Timestamp(f"{train_year}-12-31", tz="UTC")
    vy = train_year + 1
    tr = rows[known & (rows["label_end"] <= end)]
    va = rows[known & (rows["date"].dt.year == vy) & (rows["label_end"].dt.year == vy)]
    return tr, va


def main(smoke: bool = False) -> None:
    out = Path("data/reports/market_gpt_v3" + ("_smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    log(f"pre-registration: {PREREG}{'  [SMOKE: no test metrics]' if smoke else ''}")
    rd = load_research_data("price", universe="sp1500")
    dates = rd.closes.index
    if smoke:
        keep = list(rd.bars)[::25][:60]
        rd.bars = {k: rd.bars[k] for k in keep}
        rd.closes = rd.closes[keep]
        rng = np.random.default_rng(0)
        pos = np.arange(0, len(dates), 7)
        news = {
            k: (pos, np.ones(len(pos)), rng.normal(size=(len(pos), EMB_DIM)).astype(np.float16))
            for k in keep
        }
    else:
        news = news_inputs(rd, dates, progress=log)
    panel = build_panel(rd.bars, rd.benchmark, dates, news, EMB_DIM)
    rows = sample_index(panel, stride=1, membership=rd.membership)
    pre = sample_index(panel, stride=5, membership=rd.membership)
    log(
        f"{len(rows):,} daily samples, {len(pre):,} pretraining windows, "
        f"{len(panel.instruments)} stocks, news on "
        f"{(panel.news_meta[:, :, 1] == 0).mean():.1%} of stock-days"
    )

    # ---- stage 1: pretraining (one backbone per input set) ----
    tpos = dates.searchsorted(pd.Timestamp(f"{PRETRAIN_END}-12-31", tz="UTC"), side="right") - 1
    vpos = dates.searchsorted(pd.Timestamp(f"{PRETRAIN_END + 1}-12-31", tz="UTC"), side="right") - 1
    pre_tr = pre[pre["end"] + 1 <= tpos]
    pre_va = pre[(pre["date"].dt.year == PRETRAIN_END + 1) & (pre["end"] + 1 <= vpos)]
    gcfg = (
        GPTConfig(pretrain_epochs=1, warmup_steps=20)
        if smoke
        else GPTConfig(pretrain_epochs=8, patience=2)
    )
    backbones = {}
    for name, use_news in (("news", True), ("price", False)):
        path = out / f"backbone_{name}.pt"
        m = MarketGPT(EMB_DIM)
        if path.exists():
            m.load_state_dict(torch.load(path, map_location="cpu"))
            log(f"loaded pretrained {name} backbone")
        else:
            log(f"pretraining {name} backbone on {len(pre_tr):,} windows")
            m = fit_stage(m, "pretrain", panel, pre_tr, pre_va, gcfg, use_news, progress=log)
            torch.save(m.state_dict(), path)
        backbones[name] = m.cpu()

    def rcfg(**kw) -> RankConfig:
        base = (
            dict(samples_per_epoch=8_192, epochs=2, warmup_steps=5, val_every_nth_date=20)
            if smoke
            else {}
        )
        return RankConfig(**{**base, **kw})

    def trained(tag: str, backbone: str, refit: str, cfg: RankConfig) -> ScoreModel:
        path = out / f"{tag}.pt"
        m = ScoreModel(copy.deepcopy(backbones[backbone]))
        if path.exists():
            m.load_state_dict(torch.load(path, map_location="cpu"))
            return m.to(device()).eval()
        tr, va = split(rows, REFITS[refit])
        log(f"fine-tuning {tag}: {len(tr):,} training rows, {va['date'].nunique()} val dates")
        m = fit_ranker(m, panel, tr, va, cfg, backbone == "news", progress=log)
        torch.save(m.state_dict(), path)
        (out / f"{tag}.json").write_text(json.dumps({"history": cfg.history}))
        return m

    def best_val(tag: str) -> float:
        return max(h[2] for h in json.loads((out / f"{tag}.json").read_text())["history"])

    # ---- tuning on 2023 only (refit A, seed 0) ----
    for loss in ("rank", "classify"):
        trained(f"tune_{loss}_lr3e-4", "news", "A", rcfg(loss=loss, lr=3e-4))
    loss = max(("rank", "classify"), key=lambda k: best_val(f"tune_{k}_lr3e-4"))
    trained(f"tune_{loss}_lr1e-4", "news", "A", rcfg(loss=loss, lr=1e-4))
    lr = max((3e-4, 1e-4), key=lambda v: best_val(f"tune_{loss}_lr{v:.0e}".replace("e-0", "e-")))
    choice = {k: best_val(f"tune_{k}_lr3e-4") for k in ("rank", "classify")}
    log(f"2023 IC by loss: {choice}; chose loss={loss}, lr={lr:g}")
    (out / "choice.json").write_text(json.dumps({"loss": loss, "lr": lr, "val_ic": choice}))
    chosen = f"tune_{loss}_lr{lr:.0e}".replace("e-0", "e-")
    if not (out / "news_A_s0.pt").exists():  # identical config and seed: reuse, don't retrain
        for ext in (".pt", ".json"):
            (out / f"news_A_s0{ext}").write_bytes((out / f"{chosen}{ext}").read_bytes())

    # ---- ensemble + ablation, walk-forward ----
    def tag(kind: str, refit: str, seed: int) -> str:
        return f"{kind}_{refit}_s{seed}"

    for refit in REFITS:
        for seed in SEEDS:
            trained(tag("news", refit, seed), "news", refit, rcfg(loss=loss, lr=lr, seed=seed))
        trained(tag("price", refit, 0), "price", refit, rcfg(loss=loss, lr=lr, seed=0))
    log("all models trained")
    if smoke:
        log("smoke run complete (no test metrics computed)")
        return

    # ---- frozen test: every 20th trading day from 2024-01-02 ----
    test_dates = dates[dates >= pd.Timestamp("2024-01-02", tz="UTC")][::20]
    m = rows[rows["date"].isin(test_dates) & rows["fwd_ret"].notna()].reset_index(drop=True)
    refit_of = {REFITS[r] + 2: r for r in REFITS}
    m["refit"] = m["date"].dt.year.map(refit_of)
    log(f"test: {m['date'].nunique()} monthly dates, {len(m):,} stock-months")

    def zscore(s: np.ndarray) -> np.ndarray:
        g = pd.Series(s).groupby(m["date"].to_numpy())
        return ((pd.Series(s) - g.transform("mean")) / (g.transform("std") + 1e-9)).to_numpy()

    def scores(kind: str, seeds: tuple[int, ...]) -> np.ndarray:
        total = np.zeros(len(m))
        for refit in REFITS:
            ix = np.where(m["refit"] == refit)[0]
            part = m.iloc[ix]
            for s in seeds:
                model = trained(tag(kind, refit, s), kind, refit, RankConfig())
                total[ix] += predict_scores(model, panel, part, kind == "news")
        return zscore(total / len(seeds))

    ens = scores("news", SEEDS)
    news0, price0 = scores("news", (0,)), scores("price", (0,))

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

    ridge = np.zeros(len(m))
    for refit, ty in REFITS.items():
        tr, _ = split(rows, ty)
        tr = tr.sample(min(len(tr), 600_000), random_state=0)
        ix = np.where(m["refit"] == refit)[0]
        ridge[ix] = Ridge(alpha=10.0).fit(ridge_x(tr), tr["target"]).predict(ridge_x(m.iloc[ix]))

    sp500 = {f"EQ:{s}" for s in Path("data/universes/sp500.txt").read_text().split()}
    res = {
        k: eval_returns(k, m, v, rd.spread)
        for k, v in (
            ("market_gpt_v3_ensemble", ens),
            ("market_gpt_v3_price_s0", price0),
            ("ridge_baseline_v3", ridge),
            ("market_gpt_v3_news_s0", news0),
        )
    }
    for k, v in (("ensemble", ens), ("ridge", ridge)):
        for seg, mask in (
            ("S&P 500", m["instrument"].isin(sp500)),
            ("mid/small", ~m["instrument"].isin(sp500)),
        ):
            d = m[mask].assign(p=v[mask.to_numpy()])
            ic = d.groupby("date").apply(lambda g: stats.spearmanr(g["p"], g["fwd_ret"]).statistic)
            res["market_gpt_v3_ensemble" if k == "ensemble" else "ridge_baseline_v3"][
                f"segment_{seg}"
            ] = (float(ic.mean()), tstat(ic))
    diff = (
        res["market_gpt_v3_news_s0"]["_ic_series"] - res["market_gpt_v3_price_s0"]["_ic_series"]
    ).dropna()
    news_t = tstat(diff)

    print("\n=== Frozen test 2024-2026, S&P 1500, walk-forward refits (monthly) ===")
    base = res["ridge_baseline_v3"]
    for k, r in res.items():
        print(
            f"\n{k}: IC {r['ic']:+.4f} (t {r['ic_t']:+.2f}, {r['months']} months) | "
            f"beats-median accuracy {r['accuracy_vs_median']:.1%}"
        )
        for seg in ("S&P 500", "mid/small"):
            if f"segment_{seg}" in r:
                a, t = r[f"segment_{seg}"]
                print(f"  {seg}: IC {a:+.4f} (t {t:+.2f})")
        print(
            f"  top-minus-bottom decile/month: gross {r['ls_gross']:+.2%}, net tau 1 "
            f"{r['ls_net_tau1']:+.2%}, net tau 0.5 {r['ls_net_tau05']:+.2%}"
        )
        print("  per ticker: " + ", ".join(f"{t} {v:.0%}" for t, v in r["per_ticker"].items()))
    print(
        f"\nnews adds value? seed-0 IC news - price = {diff.mean():+.4f}, paired t "
        f"{news_t:+.2f} -> {'YES' if news_t >= 2 else 'no'}"
    )

    r = res["market_gpt_v3_ensemble"]
    checks = {
        "IC t >= 3.0": r["ic"] > 0 and r["ic_t"] >= 3.0,
        "net decile spread > 0 (tau 0.5)": r["ls_net_tau05"] > 0,
        "beats ridge baseline IC": r["ic"] > base["ic"],
    }
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    if all(checks.values()):
        status, verdict = "promoted", "PROMOTED: candidate direction signal for paper trading"
    elif (
        r["ic_t"] >= 2.0
        and checks["net decile spread > 0 (tau 0.5)"]
        and checks["beats ridge baseline IC"]
    ):
        status, verdict = "watchlist", "WATCHLIST: forward paper-tracking only, may not rank picks"
    else:
        status, verdict = "rejected_holdout", "REJECTED"
    print(f"VERDICT: {verdict}")

    def clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    metrics = {
        **clean(r),
        "news_vs_price_paired_t": news_t,
        "ridge_ic": base["ic"],
        "choice": json.loads((out / "choice.json").read_text()),
    }
    t1 = log_external_trial(
        "market_gpt_v3_ensemble",
        status,
        "MarketGPT v3 ensemble ranks S&P 1500 stocks by next-20-day return",
        "more data + ranking loss + walk-forward refits + 3-seed ensemble",
        metrics,
        True,
        __file__,
        notes=f"pre-registered in {PREREG}",
    )
    t2 = log_external_trial(
        "market_gpt_v3_price_s0",
        "rejected_holdout",
        "price-only ablation of MarketGPT v3",
        "news ablation",
        clean(res["market_gpt_v3_price_s0"]),
        True,
        __file__,
        notes="ablation; not eligible for promotion",
    )
    log_external_trial(
        "ridge_baseline_v3",
        "rejected_holdout",
        "ridge baseline (v3 splits)",
        "comparison",
        clean(base),
        True,
        __file__,
        notes="baseline",
    )
    print(f"trials #{t1.trial_id} (ensemble), #{t2.trial_id} (ablation)")
    if status == "promoted":
        reg = Registry.load()
        reg.record(
            SignalRecord(
                "market_gpt_v3",
                "MarketGPT v3 20-day ranking",
                "direction",
                "ic_20d",
                r["ic"],
                r["ic_t"],
                "walk-forward 2024-2026",
                f"trial #{t1.trial_id}",
            )
        )
        reg.save()
    pd.DataFrame(
        [
            {"model": k, **{a: b for a, b in clean(v).items() if not isinstance(b, dict)}}
            for k, v in res.items()
        ]
    ).to_csv(out / "results.csv", index=False)
    log("done")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
