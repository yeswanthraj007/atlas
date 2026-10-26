"""Technology-peer links from 10-K business descriptions, and peer-momentum signals.

Lee, Sun, Wang & Zhang (2019, JFE): returns of technologically linked firms predict a
firm's own future returns, because investors are slow to connect related news. Cohen &
Frazzini (2008, JF) find the same along customer-supplier links.

Here a firm's "technology fingerprint" is an embedding of Item 1 (Business) of its 10-K,
from an encoder pinned before the test period. Point-in-time: on each date a firm is
represented by its latest 10-K ACCEPTED BEFORE that date, and peers are the firms whose
fingerprints are most similar among those available then.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.marketdata.filings import read_text

_ITEM1 = re.compile(r"^\s*item\s*1\.?\s*[:\-\u2013\u2014]?\s*business\b", re.I | re.M)
_ITEM1_END = re.compile(r"^\s*item\s*(1a|2)\.?\s", re.I | re.M)


def business_section(text: str, min_chars: int = 1500, max_chars: int = 6000) -> str:
    """Item 1 text; the table of contents also matches, so keep the longest span."""
    best = ""
    for m in _ITEM1.finditer(text):
        end = _ITEM1_END.search(text, m.end())
        span = text[m.end() : end.start() if end else len(text)]
        if len(span) > len(best):
            best = span
    best = best.strip()
    return best[:max_chars] if len(best) >= min_chars else ""


def build_fingerprints(
    index: pd.DataFrame,
    cache: str | Path,
    root: str | Path = "data/sec",
    encoder=None,
    progress=print,
) -> pd.DataFrame:
    """One row per 10-K: symbol, accepted (UTC), and a unit-length embedding."""
    tenk = index[index["form"] == "10-K"].drop_duplicates("accession")
    # The cache name carries a fingerprint of the exact filing set, so adding filings
    # (e.g. a bigger universe) builds a new cache instead of silently reusing an old one.
    digest = hashlib.sha1("\n".join(sorted(tenk["accession"])).encode()).hexdigest()[:10]
    cache = Path(cache)
    cache = cache.with_name(f"{cache.stem}_{digest}{cache.suffix}")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return pd.DataFrame(
            {
                "symbol": z["symbol"],
                "accepted": pd.to_datetime(z["accepted"], utc=True),
                "emb": list(z["emb"]),
            }
        )
    texts, rows = [], []
    for r in tenk.itertuples():
        try:
            sec = business_section(read_text(root, r.cik, r.accession))
        except FileNotFoundError:
            continue
        if sec:
            texts.append(sec)
            rows.append((r.symbol, r.accepted))
    progress(f"  {len(texts):,} 10-K business sections found; encoding")
    if encoder is None:
        from atlas.ml.text_embed import Encoder

        encoder = Encoder(max_length=384, batch_size=32)
    emb = encoder.encode(texts).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    out = pd.DataFrame(
        {
            "symbol": [s for s, _ in rows],
            "accepted": pd.to_datetime([a for _, a in rows], utc=True),
            "emb": list(emb),
        }
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        symbol=out["symbol"].to_numpy(),
        accepted=out["accepted"].astype(str).to_numpy(),
        emb=np.stack(out["emb"].to_numpy()),
    )
    return out


def peer_weights(
    fp: pd.DataFrame, as_of: pd.Timestamp, k: int = 10, min_sim: float = 0.0
) -> tuple[list[str], np.ndarray]:
    """Peers on `as_of`: each firm's latest fingerprint accepted strictly before it.
    Returns (symbols, k-nearest-neighbour weight matrix, rows summing to 1, no self)."""
    avail = (
        fp[fp["accepted"] < as_of].sort_values("accepted").drop_duplicates("symbol", keep="last")
    )
    syms = avail["symbol"].tolist()
    if len(syms) <= k:
        return syms, np.zeros((len(syms), len(syms)))
    e = np.stack(avail["emb"].to_numpy())
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    w = np.zeros_like(sim)
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.arange(len(syms))[:, None]
    vals = np.maximum(sim[rows, top], min_sim)
    w[rows, top] = vals
    w /= w.sum(axis=1, keepdims=True).clip(1e-12)
    return syms, w


def peer_momentum(
    fp: pd.DataFrame,
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback: int = 21,
    k: int = 10,
    prefix: str = "EQ:",
) -> pd.DataFrame:
    """On each date: similarity-weighted past `lookback`-day return of a firm's k nearest
    technology peers, minus the firm's own past return (peers moved, it has not yet)."""
    past = closes / closes.shift(lookback) - 1.0
    rows = {}
    for d in dates:
        syms, w = peer_weights(fp, d, k)
        if not syms or not w.any():
            continue
        cols = [prefix + s for s in syms]
        r = past.loc[d].reindex(cols).to_numpy(float)
        ok = np.isfinite(r)
        w_ok = np.where(ok[None, :], w, 0.0)
        denom = w_ok.sum(axis=1)
        peer_r = np.where(
            denom > 0, (w_ok @ np.nan_to_num(r)) / np.where(denom > 0, denom, 1), np.nan
        )
        rows[d] = pd.Series(peer_r - r, index=cols)
    return pd.DataFrame(rows).T
