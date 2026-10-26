"""Which change stops v3 fine-tuning from memorising? Exploration on 2023 only (refit A,
seed 0, the saved width-128 news backbone). No 2024+ data enters any metric.

Observed in the stopped v3 run: training correlation rose to ~0.38 while 2023 IC stayed
negative (best -0.019). Hypotheses: (1) the market-return input fingerprints each date,
(2) daily samples overlap heavily, (3) too little regularisation.

    HF_HUB_OFFLINE=1 uv run python scripts/explore_v3_fixes.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from market_gpt_v3 import split
from market_transformer_v2 import EMB_DIM, news_inputs

from atlas.ml.market_gpt import (
    MarketGPT,
    RankConfig,
    ScoreModel,
    build_panel,
    fit_ranker,
    sample_index,
)
from atlas.research import load_research_data

VARIANTS = {
    "drop_market": dict(drop_market=True),
    "stride5": dict(stride=5),
    "drop_market+stride5": dict(drop_market=True, stride=5),
    "drop_market+stride5+reg": dict(drop_market=True, stride=5, weight_decay=0.2, dropout=0.3),
}


SEED_CHECK = {f"drop_market_seed{k}": dict(drop_market=True, seed=k) for k in (1, 2)}


def main() -> None:
    variants = SEED_CHECK if "--seeds" in sys.argv else VARIANTS
    rd = load_research_data("price", universe="sp1500")
    dates = rd.closes.index
    news = news_inputs(rd, dates, extra_roots=("data/news_midsmall",))
    panel = build_panel(rd.bars, rd.benchmark, dates, news, EMB_DIM)
    rows = sample_index(panel, stride=1, membership=rd.membership)
    tr, va = split(rows, 2022)
    backbone = MarketGPT(EMB_DIM, width=128)
    backbone.load_state_dict(
        torch.load("data/reports/market_gpt_v3/backbone_news_w128.pt", map_location="cpu")
    )
    out = {}
    for name, kw in variants.items():
        kw = dict(kw)
        dropout = kw.pop("dropout", 0.0)
        cfg = RankConfig(samples_per_epoch=200_000, epochs=3, patience=3, **kw)
        print(f"\n== {name} ==", flush=True)
        m = ScoreModel(copy.deepcopy(backbone), dropout=dropout)
        fit_ranker(m, panel, tr, va, cfg, True)
        out[name] = cfg.history
        best = max(h[2] for h in cfg.history)
        print(
            f"   best 2023 IC {best:+.4f}; final train corr {1 - cfg.history[-1][1]:.3f}",
            flush=True,
        )
    print("\nbaseline (stopped v3 run, no fix): best 2023 IC -0.0190, train corr up to 0.38")
    tag = "_seeds" if "--seeds" in sys.argv else ""
    Path(f"data/reports/market_gpt_v3/explore_fixes{tag}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
