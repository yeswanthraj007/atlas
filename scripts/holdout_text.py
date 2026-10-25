"""Filing-text change signal ("Lazy Prices"), tested with no fitting at all.

    uv run python scripts/holdout_text.py

The signal's direction comes from the published paper (filings that barely change ->
better future returns); the paper's sample ends before ours begins, so 2016-2026 is out
of sample for the research itself, and 2025-2026 is our frozen holdout.
Scored at 1-month and 3-month horizons on non-overlapping dates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from atlas.factors.text import TEXT_FACTORS, compute_similarities, text_factor_frames
from atlas.marketdata.filings import load_index
from atlas.research import load_research_data
from atlas.validation.signal import information_coefficient, summarize_ic

SIMS = Path("data/sec/text_similarity.parquet")
OUT = Path("data/reports/text_signal")


def deciles_spread(sig: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Top decile (most similar) minus bottom decile (biggest changers), per date."""
    out = {}
    for d in sig.index.intersection(fwd.index):
        both = pd.concat([sig.loc[d], fwd.loc[d]], axis=1, keys=["s", "r"]).dropna()
        if len(both) < 50:
            continue
        q = pd.qcut(both["s"].rank(method="first"), 10, labels=False)
        out[d] = both["r"][q == 9].mean() - both["r"][q == 0].mean()
    return pd.Series(out, dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if SIMS.exists():
        sims = pd.read_parquet(SIMS)
    else:
        sims = compute_similarities(load_index())
        sims.to_parquet(SIMS)
    print(
        f"{len(sims):,} filing comparisons, {sims['symbol'].nunique()} companies | "
        f"median cosine similarity {sims['text_sim_cosine'].median():.3f}"
    )

    rd = load_research_data("price")
    closes, members = rd.closes, rd.membership
    frames = text_factor_frames(sims, closes.index)
    idx = closes.index
    periods = {
        "2016-2024": (pd.Timestamp("2016-06-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
        "2025-2026 (frozen holdout)": (pd.Timestamp("2025-01-01", tz="UTC"), idx[-1]),
    }
    rows = []
    for horizon in (21, 63):
        fwd_all = closes.shift(-horizon) / closes - 1.0
        for pname, (a, b) in periods.items():
            dates = idx[(idx >= a) & (idx <= b)][::horizon]
            dates = dates[idx.get_indexer(dates) + horizon < len(idx)]
            mask = members.reindex(index=dates, columns=closes.columns).fillna(False)
            fwd = fwd_all.reindex(dates).where(mask)
            for name in TEXT_FACTORS:
                sig = frames[name].reindex(index=dates, columns=closes.columns).where(mask)
                ic = information_coefficient(sig, fwd)
                if len(ic.dropna()) < 3:
                    continue
                s = summarize_ic(ic)
                sp = deciles_spread(sig, fwd)
                cov = float(sig.notna().sum(axis=1).median())
                rows.append(
                    {
                        "horizon": f"{horizon}d",
                        "period": pname,
                        "signal": name,
                        "dates": s.n_dates,
                        "stocks": int(cov),
                        "mean_ic": s.mean_ic,
                        "ic_t": s.t_stat,
                        "ic_positive": s.hit_rate,
                        "spread": sp.mean(),
                        "spread_t": sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))
                        if len(sp) > 2
                        else np.nan,
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "results.csv", index=False)
    show = table.copy()
    for c, f in {
        "mean_ic": "{:+.3f}",
        "ic_t": "{:+.2f}",
        "ic_positive": "{:.0%}",
        "spread": "{:+.2%}",
        "spread_t": "{:+.2f}",
    }.items():
        show[c] = show[c].map(f.format)
    print(
        "\nspread = return of the 10% most-similar filers minus the 10% biggest changers, "
        "per period\n"
    )
    print(show.to_string(index=False))
    print(
        "\n|t| >= 2 needed to call anything real; several signals x horizons are tested, "
        "so look for consistency, not one lucky number."
    )


if __name__ == "__main__":
    main()
