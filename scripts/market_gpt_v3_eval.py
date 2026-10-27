"""Score the frozen MarketGPT v3 models exactly as pre-registered
(research/preregistered/2026-10-26-market-gpt-v3.md), plus the Amendment 3 descriptive
metrics. Loads the saved models; trains nothing.

Why this exists: the main script (market_gpt_v3.py) averages the seeds' raw scores and
z-scores once, while the pre-registration says each seed is z-scored within each date and
then averaged. This script implements the pre-registered rule; it is the one the verdict
uses. Both numbers are reported.

    HF_HUB_OFFLINE=1 uv run python scripts/market_gpt_v3_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from market_gpt_v3 import REFITS, SEEDS
from market_transformer_v2 import EMB_DIM, eval_returns, news_inputs, tstat

from atlas.backtest.periodic import decile_aims, simulate
from atlas.ml.market_gpt import MarketGPT, ScoreModel, build_panel, predict_scores, sample_index
from atlas.ml.text_embed import device
from atlas.research import load_research_data

OUT = Path("data/reports/market_gpt_v3")


def main() -> None:
    choice = json.loads((OUT / "choice.json").read_text())
    width = int(choice["width"])
    rd = load_research_data("price", universe="sp1500")
    dates = rd.closes.index
    news = news_inputs(rd, dates, extra_roots=("data/news_midsmall",))
    panel = build_panel(rd.bars, rd.benchmark, dates, news, EMB_DIM)
    rows = sample_index(panel, stride=1, membership=rd.membership)
    test_dates = dates[dates >= pd.Timestamp("2024-01-02", tz="UTC")][::20]
    m = rows[rows["date"].isin(test_dates) & rows["fwd_ret"].notna()].reset_index(drop=True)
    m["refit"] = m["date"].dt.year.map({REFITS[r] + 2: r for r in REFITS})
    groups = m["date"].to_numpy()

    def z(s: np.ndarray) -> np.ndarray:
        g = pd.Series(s).groupby(groups)
        return ((pd.Series(s) - g.transform("mean")) / (g.transform("std") + 1e-9)).to_numpy()

    def raw(kind: str, seed: int) -> np.ndarray:
        out = np.zeros(len(m))
        for refit in REFITS:
            ix = np.where(m["refit"] == refit)[0]
            model = ScoreModel(MarketGPT(EMB_DIM, width=width))
            model.load_state_dict(
                torch.load(OUT / f"{kind}_{refit}_s{seed}.pt", map_location="cpu")
            )
            model = model.to(device()).eval()
            out[ix] = predict_scores(model, panel, m.iloc[ix], kind == "news", drop_market=True)
        return out

    per_seed = {s: raw("news", s) for s in SEEDS}
    ens_prereg = np.mean([z(v) for v in per_seed.values()], axis=0)  # pre-registered rule
    ens_main = z(np.mean(list(per_seed.values()), axis=0))  # what the main script did
    r = eval_returns("market_gpt_v3_ensemble (pre-registered averaging)", m, ens_prereg, rd.spread)
    r_main = eval_returns("ensemble (main-script averaging)", m, ens_main, rd.spread)
    print("=== MarketGPT v3, frozen test 2024-2026, S&P 1500 ===")
    for x in (r, r_main):
        print(
            f"{x['model']}: IC {x['ic']:+.4f} (t {x['ic_t']:+.2f}, {x['months']} months), "
            f"net decile spread tau 0.5 {x['ls_net_tau05']:+.2%}/month"
        )

    # --- Amendment 3: descriptive metrics (reported only) ---
    d = m.assign(p=ens_prereg)
    pred = d.pivot(index="date", columns="instrument", values="p")
    fwd = d.pivot(index="date", columns="instrument", values="fwd_ret").fillna(0.0)
    cost = rd.spread.reindex(index=pred.index, columns=pred.columns)
    net = simulate(decile_aims(pred), fwd, cost, 0.5).net
    wins = int((net > 0).sum())
    p_win = stats.binomtest(wins, len(net), 0.5, alternative="greater").pvalue
    print(
        f"\nmonthly win rate (net top-minus-bottom decile > 0): {wins}/{len(net)} = "
        f"{wins / len(net):.0%} (binomial p = {p_win:.3f} vs 50%)"
    )
    med = d.groupby("date")["fwd_ret"].transform("median")
    rank = d.groupby("date")["p"].rank(pct=True)
    right = (d["p"] > d.groupby("date")["p"].transform("median")) == (d["fwd_ret"] > med)
    print(f"beats-median accuracy, all stocks: {right.mean():.1%}")
    for q in (0.05, 0.10):
        top, bot = rank >= 1 - q, rank <= q
        acc_top = (d.loc[top, "fwd_ret"] > med[top]).mean()
        acc_bot = (d.loc[bot, "fwd_ret"] <= med[bot]).mean()
        both = (top.sum() * acc_top + bot.sum() * acc_bot) / (top.sum() + bot.sum())
        print(
            f"  most confident {q:.0%} each side: top {acc_top:.1%} beat the median, "
            f"bottom {acc_bot:.1%} lagged it -> {both:.1%} right"
        )
    top10 = rank >= 0.9
    lo = d[top10].groupby("date")["fwd_ret"].mean() > d.groupby("date")["fwd_ret"].mean()
    print(
        f"long-only: top decile beat the equal-weight average in {int(lo.sum())}/{len(lo)} "
        f"months ({lo.mean():.0%})"
    )
    seed_ic = {
        s: float(
            d.assign(p=z(v))
            .groupby("date")
            .apply(lambda g: stats.spearmanr(g["p"], g["fwd_ret"]).statistic)
            .mean()
        )
        for s, v in per_seed.items()
    }
    print(f"single-seed test ICs: {seed_ic}")
    year_ic = d.groupby("date").apply(lambda g: stats.spearmanr(g["p"], g["fwd_ret"]).statistic)
    print(
        "IC by year: "
        + ", ".join(f"{y}: {v:+.4f}" for y, v in year_ic.groupby(year_ic.index.year).mean().items())
    )
    (OUT / "eval_prereg.json").write_text(
        json.dumps(
            {
                "ic": r["ic"],
                "ic_t": r["ic_t"],
                "ls_net_tau05": r["ls_net_tau05"],
                "main_script_ic": r_main["ic"],
                "main_script_ic_t": r_main["ic_t"],
                "monthly_win_rate": wins / len(net),
                "win_p": p_win,
                "seed_ic": seed_ic,
                "accuracy_all": float(right.mean()),
            },
            indent=2,
        )
    )
    print(f"\nnote: verdict uses the pre-registered averaging; tstat={tstat(r['_ic_series']):+.2f}")


if __name__ == "__main__":
    main()
