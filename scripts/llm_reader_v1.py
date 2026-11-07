# ruff: noqa: E501
"""Language-model reader of earnings releases v1, exactly as pre-registered in
research/preregistered/2026-11-07-llm-reader-v1.md.

    HF_HUB_OFFLINE=1 uv run python scripts/llm_reader_v1.py            # full run (hours)
    HF_HUB_OFFLINE=1 uv run python scripts/llm_reader_v1.py --smoke    # plumbing check, never touches 2024+

Stages are cached under data/llm_reader/, so an interrupted run resumes (at epoch granularity).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from atlas.factors.links import company_aliases
from atlas.lab.trials import log_external_trial
from atlas.marketdata.earnings_releases import read_release
from atlas.ml.llm_reader import (
    MODEL,
    REVISION,
    ReaderConfig,
    drift_labels,
    gaussian_rank_by_group,
    load_reader,
    predict_reader,
    prepare_text,
    tokenize,
    train_reader,
)
from atlas.ml.risk_events import build_events
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import load_research_data

PREREG = "research/preregistered/2026-11-07-llm-reader-v1.md"
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-09-30", tz="UTC")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def build_dataset(smoke: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    out = Path("data/llm_reader")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ("dataset_smoke.parquet" if smoke else "dataset.parquet")
    toks = out / ("tokens_smoke.npz" if smoke else "tokens.npz")
    if cache.exists() and toks.exists():
        z = np.load(toks)
        return pd.read_parquet(cache), z["ids"], z["mask"]
    rd = load_research_data("price", universe="sp1500")
    idx, closes = rd.closes.index, rd.closes
    ev = build_events(closes, rd.benchmark, pd.read_parquet("data/sec/earnings_index.parquet"))
    mem = rd.membership.reindex(index=idx, columns=closes.columns).fillna(False).to_numpy()
    ev = ev[mem[ev["p"].to_numpy(), ev["stock"].to_numpy()]].reset_index(drop=True)
    ev["drift"] = drift_labels(closes, rd.benchmark, ev["d0"].to_numpy(), ev["stock"].to_numpy())
    ev["label_date"] = [
        idx[min(d + 20, len(idx) - 1)] if np.isfinite(x) else pd.NaT
        for d, x in zip(ev["d0"], ev["drift"], strict=True)
    ]
    ev = ev[ev["react"].notna() & ev["y_e1"].notna()].reset_index(drop=True)
    q = ev["accepted"].dt.year * 10 + ev["accepted"].dt.quarter
    ev["quarter"] = q.to_numpy()
    ok = ev["drift"].notna()
    ev["y2"] = np.nan
    ev.loc[ok, "y2"] = gaussian_rank_by_group(ev.loc[ok, "drift"], ev.loc[ok, "quarter"])
    ev["y1"] = (ev["react"] > 0).astype(float)
    ev["y3"] = ev["y_e1"]
    ev["month"] = ev["accepted"].dt.strftime("%Y-%m")
    ev["cost"] = rd.spread.to_numpy()[ev["p"].to_numpy(), ev["stock"].to_numpy()]
    ev["cost"] = ev["cost"].fillna(np.nanmedian(ev["cost"]))
    split = np.where(
        ok & (ev["label_date"] <= TRAIN_END), "train",
        np.where(ok & (ev["accepted"].dt.year == 2023) & (ev["label_date"] <= VAL_END), "val",
                 np.where(ok & (ev["accepted"] >= TEST_START) & (ev["accepted"] <= TEST_END), "test", "none")),
    )  # fmt: skip
    ev["split"] = split
    ev = ev[ev["split"] != "none"].reset_index(drop=True)
    if smoke:  # small, and never any 2024+ event
        ev = pd.concat([
            ev[ev.split == "train"].sample(480, random_state=0),
            ev[ev.split == "val"].sample(160, random_state=0),
        ]).reset_index(drop=True)  # fmt: skip
    cons = pd.concat(
        [pd.read_csv(f"data/universes/{u}_constituents.csv") for u in ("sp500", "sp400", "sp600")]
    ).drop_duplicates("Symbol")
    aliases = company_aliases(cons)
    texts, keep = [], []
    for i, r in enumerate(ev.itertuples()):
        try:
            raw = read_release("data/sec", r.cik, r.accession)
        except Exception:
            continue
        sym = r.instrument.replace("EQ:", "")
        texts.append(prepare_text(raw, aliases.get(sym, []), sym))
        keep.append(i)
        if len(keep) % 10000 == 0:
            log(f"  read {len(keep):,} releases")
    ev = ev.iloc[keep].reset_index(drop=True)
    ev["text"] = texts
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    ids, mask = tokenize(texts, tok)
    ev.drop(columns=["text"]).assign(text=texts).to_parquet(cache)
    np.savez(toks, ids=ids, mask=mask)
    return ev, ids, mask


def monthly_ic(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    d = df[["month", "drift"]].assign(s=score)
    return (
        d.groupby("month")
        .apply(lambda g: stats.spearmanr(g["s"], g["drift"]).statistic if len(g) >= 30 else np.nan)
        .dropna()
    )


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def decile_spread(df: pd.DataFrame, score: np.ndarray) -> tuple[float, float, float]:
    d = df[["month", "drift", "cost"]].assign(s=score)
    gross, net = [], []
    for _, g in d.groupby("month"):
        if len(g) < 30:
            continue
        k = max(3, len(g) // 10)
        g = g.sort_values("s")
        lo, hi = g.iloc[:k], g.iloc[-k:]
        gr = hi["drift"].mean() - lo["drift"].mean()
        gross.append(gr)
        net.append(gr - hi["cost"].mean() - lo["cost"].mean())
    return float(np.mean(gross)), float(np.mean(net)), tstat(pd.Series(net))


def metrics(df: pd.DataFrame, out: np.ndarray) -> dict:
    """out columns: T1 score, T2 score, T3 score (logit or regression value)."""
    ic = monthly_ic(df, out[:, 1])
    gross, net, net_t = decile_spread(df, out[:, 1])
    return {
        "t1_acc": float(((out[:, 0] > 0) == (df["y1"] > 0.5)).mean()),
        "t1_auc": float(roc_auc_score(df["y1"], out[:, 0])),
        "t3_acc": float(((out[:, 2] > 0) == (df["y3"] > 0.5)).mean()),
        "t3_auc": float(roc_auc_score(df["y3"], out[:, 2])),
        "t2_ic": float(ic.mean()), "t2_t": tstat(ic), "t2_months": len(ic),
        "spread_gross": gross, "spread_net": net, "spread_net_t": net_t, "n": len(df),
        "_ic": ic,
    }  # fmt: skip


def tfidf_baseline(df: pd.DataFrame, texts: pd.Series) -> np.ndarray:
    tr = (df["split"] == "train").to_numpy()
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, sublinear_tf=True, min_df=3)
    x = vec.fit_transform(texts[tr])
    models = {
        "y1": LogisticRegression(C=1.0, max_iter=1000).fit(x, df.loc[tr, "y1"]),
        "y3": LogisticRegression(C=1.0, max_iter=1000).fit(x, df.loc[tr, "y3"]),
        "y2": Ridge(alpha=1.0).fit(x, df.loc[tr, "y2"]),
    }
    xa = vec.transform(texts)
    return np.column_stack(
        [
            models["y1"].decision_function(xa),
            models["y2"].predict(xa),
            models["y3"].decision_function(xa),
        ]
    )


def main(smoke: bool) -> None:
    log(f"pre-registration: {PREREG}{'  [SMOKE: val only, nothing logged]' if smoke else ''}")
    ev, ids, mask = build_dataset(smoke)
    texts = pd.read_parquet(
        "data/llm_reader/" + ("dataset_smoke.parquet" if smoke else "dataset.parquet"),
        columns=["text"],
    )["text"]
    counts = ev["split"].value_counts().to_dict()
    log(f"{len(ev):,} events: {counts}")

    def split(name: str) -> dict:
        m = (ev["split"] == name).to_numpy()
        return {
            "ids": ids[m],
            "mask": mask[m],
            "y1": ev.loc[m, "y1"].to_numpy(),
            "y2": ev.loc[m, "y2"].to_numpy(),
            "y3": ev.loc[m, "y3"].to_numpy(),
        }

    eval_name = "val" if smoke else "test"
    base_all = tfidf_baseline(ev, texts)
    sel = (ev["split"] == eval_name).to_numpy()
    base = metrics(ev[sel], base_all[sel])
    log(
        f"TF-IDF baseline on {eval_name}: T1 acc {base['t1_acc']:.1%} (AUC {base['t1_auc']:.3f}) | T3 acc {base['t3_acc']:.1%} (AUC {base['t3_auc']:.3f}) | T2 IC {base['t2_ic']:+.4f} (t {base['t2_t']:+.2f})"
    )

    ckpt = Path("data/llm_reader/ckpt" + ("_smoke" if smoke else ""))
    cfg = ReaderConfig(epochs=1, batch=16) if smoke else ReaderConfig()
    model = load_reader()
    tr, va = split("train"), split("val")
    if smoke:
        tr = {k: v[:256] for k, v in tr.items()}
    log(f"training: {len(tr['ids']):,} train, {len(va['ids']):,} validation, {cfg.epochs} epoch(s)")
    model = train_reader(model, tr, va, cfg, ckpt, progress=log)
    pred = predict_reader(model, split(eval_name))
    llm = metrics(ev[sel], pred)
    pd.DataFrame(pred, columns=["t1", "t2", "t3"]).assign(
        month=ev.loc[sel, "month"].to_numpy()
    ).to_parquet(f"data/llm_reader/pred_{eval_name}.parquet")

    print(
        f"\n=== {'VALIDATION (smoke)' if smoke else 'FROZEN TEST 2024-2026'}: {int(sel.sum()):,} events ==="
    )
    for name, m in (("LLM reader (RoBERTa)", llm), ("TF-IDF baseline", base)):
        print(f"\n{name}\n  T1 reaction sign (nowcast, not tradable): accuracy {m['t1_acc']:.1%}, AUC {m['t1_auc']:.3f}"
              f"\n  T3 move size vs quarter median: accuracy {m['t3_acc']:.1%}, AUC {m['t3_auc']:.3f}"
              f"\n  T2 post-reaction drift: IC {m['t2_ic']:+.4f} (t {m['t2_t']:+.2f}, {m['t2_months']} months) | top-minus-bottom decile {m['spread_gross']:+.2%} gross, {m['spread_net']:+.2%} net (t {m['spread_net_t']:+.2f})")  # fmt: skip
    ic_diff = llm["_ic"].sub(base["_ic"], fill_value=np.nan).dropna()
    pm = ev[sel].assign(
        h=((pred[:, 2] > 0) == (ev.loc[sel, "y3"] > 0.5)).to_numpy(),
        b=((base_all[sel][:, 2] > 0) == (ev.loc[sel, "y3"] > 0.5)).to_numpy(),
    )
    t3_diff = (pm.groupby("month")["h"].mean() - pm.groupby("month")["b"].mean()).dropna()
    print(
        f"\nPaired monthly t: LLM minus TF-IDF on T2 IC {tstat(ic_diff):+.2f}; on T3 accuracy {tstat(t3_diff):+.2f}"
    )
    if smoke:
        log("smoke run complete (validation only)")
        return

    checks = {"T2 IC > 0 with t >= 3": llm["t2_ic"] > 0 and llm["t2_t"] >= 3.0,
              "net decile spread > 0": llm["spread_net"] > 0,
              "IC above the TF-IDF baseline's": llm["t2_ic"] > base["t2_ic"]}  # fmt: skip
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    promoted = all(checks.values())
    sec = llm["t3_acc"] >= 0.65 and tstat(t3_diff) >= 2.0
    print(
        f"  {'PASS' if sec else 'FAIL'}  secondary: T3 accuracy >= 65% and above TF-IDF with paired t >= 2"
    )
    print(
        f"VERDICT (tradable criterion): {'PROMOTED to paper-trading candidate' if promoted else 'rejected'}"
    )
    reg = Registry.load()
    for tag, status in (
        ("t2", "promoted" if promoted else "rejected_holdout"),
        ("t3", "rejected_holdout"),
        ("t1", "rejected_holdout"),
    ):
        clean = {k: v for k, v in llm.items() if not k.startswith("_")}
        t = log_external_trial(f"llm_reader_{tag}", status, f"a RoBERTa reader of earnings releases predicts {tag.upper()}", "fine-tuned language model on text only, chronologically clean", {**clean, "baseline": {k: v for k, v in base.items() if not k.startswith("_")}, "paired_t_ic": tstat(ic_diff), "paired_t_t3": tstat(t3_diff)}, True, __file__, notes=f"pre-registered in {PREREG}")  # fmt: skip
        log(f"trial #{t.trial_id}: llm_reader_{tag}: {status}")
        if tag == "t2" and promoted:
            reg.record(
                SignalRecord(
                    "llm_reader_drift",
                    "RoBERTa earnings-release reader, 20-day drift",
                    "direction",
                    "ic_20d",
                    llm["t2_ic"],
                    llm["t2_t"],
                    "frozen 2024-2026",
                    f"trial #{t.trial_id}",
                )
            )
    reg.save()


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
