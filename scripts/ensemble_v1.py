"""Weak-signal ensemble v1, exactly as pre-registered in
research/preregistered/2026-10-26-ensemble-v1.md (read that first; this script must not
deviate from it, and any deviation is printed as one).

    uv run python scripts/ensemble_v1.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from atlas.backtest.periodic import decile_aims, simulate
from atlas.factors.links import link_matrix
from atlas.factors.panel import price_factor_frames
from atlas.factors.peers import build_fingerprints, peer_momentum
from atlas.lab.trials import log_external_trial
from atlas.marketdata.earnings_releases import load_earnings_index, read_release
from atlas.marketdata.filings import load_index
from atlas.ml.news_model import NewsSentimentModel, identity_tokens, news_day_index
from atlas.ml.text_embed import ENCODER_REVISION
from atlas.research import load_research_data
from atlas.validation.overfit import min_detectable_ic, pbo_cscv
from atlas.validation.stats import alpha_beta

PREREG = "research/preregistered/2026-10-26-ensemble-v1.md"
OUT = Path("data/reports/ensemble_v1")
H = 21
ALPHAS = (1.0, 10.0, 100.0)
TAUS = (0.25, 0.5, 1.0)
PRIMARY = ("ridge", 10.0, 0.5)
INPUTS = [
    "reversal_1m",
    "mom_12_1",
    "low_vol",
    "tech_peers",
    "customer_mom",
    "earnings_text",
    "gross_profitability",
    "earnings_yield",
    "book_to_price",
]


def centred_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True) - 0.5


def rolling_earnings_scores(closes: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    """Score per stock-day from the latest earnings release (valid 63 days). The model used
    for a release in year Y was trained only on releases whose 21-day outcome was known
    before Jan 1 of year Y."""
    idx = closes.index
    syms = {c.split(":")[1]: c for c in closes.columns}
    ev = load_earnings_index()
    ev = ev[ev["symbol"].isin(syms)].drop_duplicates("accession").copy()
    ev["text"] = [
        read_release("data/sec", r.cik, r.accession)[:4000].lower() for r in ev.itertuples()
    ]
    ev["d0"] = news_day_index(pd.to_datetime(ev["accepted"], utc=True), idx)
    ev = ev[(ev["d0"] >= 1) & (ev["d0"] + 1 < len(idx))].copy()
    c = closes.to_numpy()
    j = ev["symbol"].map(lambda s: closes.columns.get_loc(syms[s])).to_numpy()
    d0 = ev["d0"].to_numpy()
    sc = spy.reindex(idx).to_numpy()
    ev["reaction"] = c[d0 + 1, j] / c[d0 - 1, j] - sc[d0 + 1] / sc[d0 - 1]
    ev["year"] = idx[d0].year
    ev["label_end"] = idx[np.minimum(d0 + H, len(idx) - 1)]
    out = pd.DataFrame(np.nan, index=idx, columns=closes.columns)
    for year in sorted(ev["year"].unique()):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        train = ev[(ev["label_end"] < start) & ev["reaction"].notna()]
        test = ev[ev["year"] == year]
        if len(train) < 2000 or test.empty:
            continue
        model = NewsSentimentModel(stop_words=identity_tokens(train, min_count=20), min_df=10)
        model.fit(train["text"], train["reaction"])
        for r, s in zip(test.itertuples(), model.score(test["text"]), strict=True):
            first = r.d0 + 1  # tradeable from the next session
            out.iloc[first : first + 63, closes.columns.get_loc(syms[r.symbol])] = s
    return out


def customer_frame(closes: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    links = pd.read_parquet("data/sec/links.parquet")
    past = closes / closes.shift(21) - 1.0
    rows = {}
    for d in dates:
        m = link_matrix(links, d, "customer", min_mentions=2)
        sig = {}
        for src, customers in m.items():
            r = np.array([past.loc[d].get(f"EQ:{t}", np.nan) for t in customers])
            w = np.array(list(customers.values()), dtype=float)
            ok = np.isfinite(r)
            if ok.any():
                sig[f"EQ:{src}"] = float((w[ok] * r[ok]).sum() / w[ok].sum())
        rows[d] = pd.Series(sig, dtype=float)
    return pd.DataFrame(rows).T.reindex(columns=closes.columns)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pre-registration: {PREREG}")
    rd = load_research_data("all", universe="sp1500")
    closes, idx = rd.closes, rd.closes.index
    dates = idx[(idx >= pd.Timestamp("2016-06-01", tz="UTC"))][::H]
    dates = dates[idx.get_indexer(dates) + H < len(idx)]
    print(
        f"{closes.shape[1]} stocks | {len(dates)} monthly dates "
        f"{dates[0].date()}..{dates[-1].date()}"
    )

    pf = price_factor_frames(closes)
    raw = {k: pf[k] for k in ("reversal_1m", "mom_12_1", "low_vol")}
    fp = build_fingerprints(
        load_index(), Path(f"data/sec/business_fingerprints_{ENCODER_REVISION[:10]}.npz")
    )
    raw["tech_peers"] = peer_momentum(fp, closes, dates, lookback=21, k=10)
    raw["customer_mom"] = customer_frame(closes, dates)
    raw["earnings_text"] = rolling_earnings_scores(closes, rd.benchmark)
    for k in ("gross_profitability", "earnings_yield", "book_to_price"):
        raw[k] = rd.factors[k]
    members = rd.membership.reindex(index=dates, columns=closes.columns).fillna(False)
    feats = {
        k: centred_rank(v.reindex(index=dates, columns=closes.columns).where(members))
        for k, v in raw.items()
    }
    coverage = {k: float(v.notna().sum(axis=1).median()) for k, v in feats.items()}
    print("median stocks covered per date:", json.dumps({k: int(v) for k, v in coverage.items()}))

    fwd = (closes.shift(-H) / closes - 1.0).reindex(dates).where(members)
    target = centred_rank(fwd)
    long = pd.DataFrame({k: v.stack(future_stack=True) for k, v in feats.items()}).fillna(0.0)
    long["y"] = target.stack(future_stack=True)
    long = long[members.stack(future_stack=True).reindex(long.index).fillna(False).to_numpy()]
    label_end = pd.Series(idx[idx.get_indexer(dates) + H], index=dates)

    preds: dict[tuple, pd.DataFrame] = {}
    for year in range(2018, idx[-1].year + 1):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        tr_dates = label_end.index[label_end < start]
        te_dates = dates[(dates >= start) & (dates < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))]
        lvl = long.index.get_level_values(0)
        tr = long[lvl.isin(tr_dates) & long["y"].notna()]
        te = long[lvl.isin(te_dates)]
        if te.empty:
            continue
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(tr[INPUTS], tr["y"])
            p = pd.Series(m.predict(te[INPUTS]), index=te.index).unstack()
            preds.setdefault(("ridge", a), []).append(p)
        eq = te[INPUTS].mean(axis=1).unstack()
        preds.setdefault(("equal", 0.0), []).append(eq)
    preds = {k: pd.concat(v).sort_index() for k, v in preds.items()}

    fwd_d = fwd.reindex(index=next(iter(preds.values())).index)
    cost = rd.spread.reindex(index=fwd_d.index, columns=closes.columns)
    spy = rd.benchmark.reindex(idx)
    spy_fwd = (spy.shift(-H) / spy - 1.0).reindex(fwd_d.index)
    results, nets, lo_nets = {}, {}, {}
    for (kind, a), p in preds.items():
        ic = p.corrwith(fwd_d, axis=1, method="spearman").dropna()
        for tau in TAUS:
            ls = simulate(decile_aims(p), fwd_d.fillna(0.0), cost, tau)
            lo = simulate(decile_aims(p, long_short=False, top_n=50), fwd_d.fillna(0.0), cost, tau)
            key = f"{kind}_a{a:g}_tau{tau:g}"
            net = ls.net
            nets[key] = net
            lo_ex = lo.net - spy_fwd.reindex(lo.net.index)
            lo_nets[key] = lo_ex
            ab = alpha_beta(lo.net.to_numpy(), spy_fwd.reindex(lo.net.index).to_numpy(), 12)
            recent = net[net.index >= pd.Timestamp("2024-01-01", tz="UTC")]
            t_net = float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net))))
            results[key] = {
                "ic": float(ic.mean()),
                "ic_t": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))),
                "ls_gross_month": float(ls.gross.mean()),
                "ls_cost_month": float(ls.cost.mean()),
                "ls_net_month": float(net.mean()),
                "ls_net_t": t_net,
                "ls_net_sharpe_ann": float(net.mean() / net.std(ddof=1) * np.sqrt(12)),
                "ls_net_2024_2026": float(recent.mean()),
                "turnover": float(ls.turnover.mean()),
                "lo_excess_month": float(lo_ex.mean()),
                "lo_alpha_ann": ab["alpha_ann"],
                "lo_alpha_t": ab["alpha_t"],
                "lo_beta": ab["beta"],
                "ic_by_year": {
                    str(y): round(float(v), 4) for y, v in ic.groupby(ic.index.year).mean().items()
                },
            }
    table = pd.DataFrame(results).T
    print("\n=== All 12 variants (monthly figures; net = after assumed tier costs) ===")
    cols = [
        "ic",
        "ic_t",
        "ls_gross_month",
        "ls_cost_month",
        "ls_net_month",
        "ls_net_t",
        "ls_net_sharpe_ann",
        "ls_net_2024_2026",
        "turnover",
        "lo_alpha_ann",
        "lo_alpha_t",
    ]
    print(table[cols].astype(float).round(4).to_string())

    pbo = pbo_cscv(pd.DataFrame(nets).dropna().to_numpy(), n_blocks=8)
    prim_key = f"ridge_a{PRIMARY[1]:g}_tau{PRIMARY[2]:g}"
    prim = results[prim_key]
    ic_series = preds[("ridge", PRIMARY[1])].corrwith(fwd_d, axis=1, method="spearman").dropna()
    mde = min_detectable_ic(len(ic_series), float(ic_series.std(ddof=1)))
    checks = {
        "1. net long-short t >= 3 (2018-2026)": prim["ls_net_t"] >= 3.0,
        "2. net long-short mean > 0 (2024-2026)": prim["ls_net_2024_2026"] > 0,
        "3. PBO < 0.3 across 12 variants": pbo["pbo"] < 0.3,
        "4. long-only net CAPM alpha t >= 2": prim["lo_alpha_t"] >= 2.0,
    }
    print(f"\n=== Primary specification {prim_key} (pre-registered) ===")
    print(
        f"IC {prim['ic']:+.4f} (t {prim['ic_t']:+.2f}) | minimum detectable IC with "
        f"{len(ic_series)} months: {mde:.4f}"
    )
    print(
        f"long-short: gross {prim['ls_gross_month']:+.2%}/month, cost "
        f"{prim['ls_cost_month']:.2%}, net {prim['ls_net_month']:+.2%} (t "
        f"{prim['ls_net_t']:+.2f}), net Sharpe {prim['ls_net_sharpe_ann']:.2f}"
    )
    print(
        f"long-only top 50: alpha {prim['lo_alpha_ann']:+.1%}/yr (t {prim['lo_alpha_t']:+.2f})"
        f", beta {prim['lo_beta']:.2f}"
    )
    print(
        f"PBO {pbo['pbo']:.2f} over {pbo['splits']} splits; mean Sharpe degradation "
        f"{pbo['mean_sharpe_degradation']:+.3f}"
    )
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    passed = all(checks.values())
    print(f"\nVERDICT: {'all pre-registered criteria met' if passed else 'not met'}")
    table.to_csv(OUT / "variants.csv")
    pd.DataFrame(nets).to_csv(OUT / "net_long_short.csv")
    trial = log_external_trial(
        "ensemble_v1",
        "promoted" if passed else "rejected_holdout",
        "Weak signals combined with shrinkage and traded with partial rebalancing earn a "
        "net-of-cost edge (S&P 1500)",
        "Kozak-Nagel-Santosh / Kelly-Malamud-Zhou shrinkage across many characteristics; "
        "Jensen-Kelly-Malamud-Pedersen and Garleanu-Pedersen cost-aware trading",
        {
            "primary": prim,
            "pbo": pbo,
            "checks": {k: bool(v) for k, v in checks.items()},
            "min_detectable_ic": mde,
            "coverage": coverage,
            "variants": 12,
        },
        True,
        __file__,
        notes=f"pre-registered in {PREREG}; 12 variants",
    )
    print(f"logged as trial #{trial.trial_id}")
    _ = itertools


if __name__ == "__main__":
    main()
