"""Engineering diagnostics for MarketGPT. Uses only data up to 2023 (the frozen 2024-2026
test is never touched).

1. Planted-signal test: add a known, learnable pattern to the real 20-day returns (a mix of
   5-day reversal and one news component), retrain from scratch, and check that the model
   recovers it on 2023. If it cannot, the pipeline has a bug.
2. Learning check: IC of the saved v2 models on training years vs 2023. Large train IC and
   zero 2023 IC = overfitting; zero on both = nothing learnable found.

    uv run python scripts/diagnose_market_gpt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from market_transformer_v2 import (
    EMB_DIM,
    OUT,
    TRAIN_END,
    VAL_YEAR,
    _load,
    news_inputs,
)

from atlas.ml.market_gpt import (
    GPTConfig,
    MarketGPT,
    build_panel,
    fit_stage,
    predict_rows,
    sample_index,
)
from atlas.research import load_research_data


def ic_by_date(rows: pd.DataFrame, score: np.ndarray, col: str = "fwd_ret") -> tuple[float, float]:
    d = rows.assign(p=score)
    ic = d.groupby("date").apply(lambda g: stats.spearmanr(g["p"], g[col]).statistic).dropna()
    return float(ic.mean()), float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))


def main() -> None:
    rd = load_research_data("price", universe="sp500")
    dates = rd.closes.index
    panel = build_panel(rd.bars, rd.benchmark, dates, news_inputs(rd, dates), EMB_DIM)
    rows = sample_index(panel, stride=5, membership=rd.membership)
    yr = rows["date"].dt.year
    known = rows["target"].notna()
    tr = rows[known & (rows["label_end"] <= TRAIN_END)].reset_index(drop=True)
    va = rows[known & (yr == VAL_YEAR) & (rows["label_end"].dt.year == VAL_YEAR)].reset_index(
        drop=True
    )
    print(f"train {len(tr):,} | 2023 validation {len(va):,} (2024+ never loaded into a metric)")

    # 2. learning check on the saved, frozen models
    print("\n== Did the saved models learn anything? (IC vs real 20-day returns) ==")
    sub = tr.sample(30_000, random_state=0).sort_values("date")
    for name, news in (("market_gpt_news", True), ("market_gpt_price", False)):
        m = _load(MarketGPT(EMB_DIM), OUT / f"{name}.pt")
        a = ic_by_date(sub, predict_rows(m, "classify", panel, sub, news))
        b = ic_by_date(va, predict_rows(m, "classify", panel, va, news))
        print(
            f"{name}: train-years IC {a[0]:+.4f} (t {a[1]:+.2f}) | 2023 IC {b[0]:+.4f} "
            f"(t {b[1]:+.2f})"
        )

    # 1. planted-signal test
    print("\n== Planted-signal test ==")
    days = lambda r: r["end"].to_numpy()[:, None] + np.arange(-4, 1)[None, :]  # noqa: E731
    for name, d in (("train", tr), ("val", va)):
        s, e = d["stock"].to_numpy(), days(d)
        rev = -np.nansum(panel.price[s[:, None], e, 0], axis=1)
        newsc = panel.news_emb[s, d["end"].to_numpy(), 0].astype(float)
        sig = (
            pd.Series(rev / 5 + 3 * newsc)
            .groupby(d["date"].to_numpy())
            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            .to_numpy()
        )
        sd = d.groupby("date")["fwd_ret"].transform("std").to_numpy()
        d["planted_ret"] = d["fwd_ret"] + 0.25 * sd * sig  # true IC of the planted part ~0.24
        med = d.groupby("date")["planted_ret"].transform("median")
        d["target"] = (d["planted_ret"] > med).astype(float)
        d["planted_sig"] = sig
        if name == "train":
            tr = d
        else:
            va = d
    best = ic_by_date(va, va["planted_sig"].to_numpy(), "planted_ret")
    print(f"ceiling: IC of the planted signal itself on 2023 = {best[0]:+.3f}")
    model = fit_stage(
        MarketGPT(EMB_DIM),
        "classify",
        panel,
        tr,
        va,
        GPTConfig(finetune_epochs=4, patience=2),
        True,
    )
    got = ic_by_date(va, predict_rows(model, "classify", panel, va, True), "planted_ret")
    print(
        f"MarketGPT recovered: 2023 IC {got[0]:+.3f} (t {got[1]:+.2f}) = "
        f"{got[0] / best[0]:.0%} of the ceiling -> "
        f"{'PIPELINE OK' if got[0] > 0.5 * best[0] else 'PIPELINE PROBLEM'}"
    )


if __name__ == "__main__":
    main()
