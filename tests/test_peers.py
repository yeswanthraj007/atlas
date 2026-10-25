import numpy as np
import pandas as pd

from atlas.factors.peers import business_section, peer_momentum, peer_weights


def test_business_section_skips_table_of_contents() -> None:
    body = "We design memory chips for data centers. " * 60
    text = (
        f"Item 1. Business ..... 4\nItem 1A. Risk Factors ..... 12\n"
        f"Item 1. Business\n{body}\nItem 1A. Risk Factors\nRisks follow."
    )
    sec = business_section(text)
    assert sec.startswith("We design memory chips") and "Risks follow" not in sec


def _fp(rows):
    return pd.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "accepted": pd.to_datetime([r[1] for r in rows], utc=True),
            "emb": [np.array(r[2], dtype=np.float32) for r in rows],
        }
    )


def test_peers_are_point_in_time_and_exclude_self() -> None:
    fp = _fp(
        [
            ("A", "2020-02-01", [1, 0]),
            ("B", "2020-02-01", [0.9, 0.1]),
            ("C", "2020-02-01", [0, 1]),
            ("D", "2021-02-01", [1, 0.01]),
        ]
    )
    syms, w = peer_weights(fp, pd.Timestamp("2020-06-01", tz="UTC"), k=1)
    assert syms == ["A", "B", "C"]  # D not filed yet
    assert w[0, 1] == 1.0 and np.allclose(np.diag(w), 0)  # A's peer is B, never itself


def test_peer_momentum_flags_laggards() -> None:
    idx = pd.bdate_range("2020-03-02", periods=40, tz="UTC")
    closes = pd.DataFrame(
        {
            "EQ:A": np.linspace(100, 100, 40),  # flat
            "EQ:B": np.linspace(100, 130, 40),  # A's peer rallies
            "EQ:C": np.linspace(100, 101, 40),
        },
        index=idx,
    )
    fp = _fp(
        [
            ("A", "2020-02-01", [1, 0]),
            ("B", "2020-02-01", [0.95, 0.05]),
            ("C", "2020-02-01", [0, 1]),
        ]
    )
    sig = peer_momentum(fp, closes, idx[-1:], lookback=21, k=1)
    assert sig.loc[idx[-1], "EQ:A"] > 0.1  # peer up, A has not moved: catch-up candidate
    assert sig.loc[idx[-1], "EQ:B"] < 0  # B already ran ahead of its peer A
