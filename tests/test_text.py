import gzip

import numpy as np
import pandas as pd
import pytest

from atlas.factors.text import (
    compute_similarities,
    cosine,
    jaccard,
    match_previous,
    text_factor_frames,
    word_counts,
)
from atlas.marketdata.filings import html_to_text, risk_factors_section


def test_similarity_measures() -> None:
    a = word_counts("the cat sat on the mat")
    assert cosine(a, a) == pytest.approx(1.0)
    assert jaccard(a, a) == 1.0
    b = word_counts("completely different words here")
    assert cosine(a, b) == 0.0 and jaccard(a, b) == 0.0
    c = word_counts("the cat sat on the rug")
    assert 0.0 < cosine(a, c) < 1.0


def test_html_cleaning_and_risk_factor_extraction() -> None:
    body = "<p>Item 1A. Risk Factors</p><p>" + "We face supply risk. " * 200 + "</p>"
    raw = (
        "<html><head><title>x</title></head><body><ix:header>hidden</ix:header>"
        "<p>Item 1A. Risk Factors .... 12</p><p>Item 2. Properties ... 30</p>"
        f"{body}<p>Item 1B. Unresolved Staff Comments</p></body></html>"
    )
    text = html_to_text(raw)
    assert "hidden" not in text
    rf = risk_factors_section(text)
    assert rf.startswith("We face supply risk") and "Unresolved" not in rf


def _index(rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["symbol", "cik", "form", "accession", "filed", "accepted", "period", "document"],
    )


def test_matching_compares_same_form_one_year_apart() -> None:
    idx = _index(
        [
            ["A", 1, "10-K", "k1", "2023-02-01", "2023-02-01T21:00:00Z", "2022-12-31", "d"],
            ["A", 1, "10-K", "k2", "2024-02-01", "2024-02-01T21:00:00Z", "2023-12-31", "d"],
            ["A", 1, "10-Q", "q1", "2023-05-01", "2023-05-01T21:00:00Z", "2023-03-31", "d"],
            ["A", 1, "10-Q", "q2", "2023-08-01", "2023-08-01T21:00:00Z", "2023-06-30", "d"],
            ["A", 1, "10-Q", "q5", "2024-05-01", "2024-05-01T21:00:00Z", "2024-03-31", "d"],
        ]
    )
    m = match_previous(idx).set_index("accession")["prev_accession"]
    assert m["k2"] == "k1" and m["q5"] == "q1"  # same quarter, prior year
    assert m["k1"] is None and m["q2"] is None


def test_signal_is_point_in_time(tmp_path) -> None:
    idx = _index(
        [
            ["A", 1, "10-K", "k1", "2023-02-01", "2023-02-01T21:00:00Z", "2022-12-31", "d"],
            ["A", 1, "10-K", "k2", "2024-02-01", "2024-02-01T21:30:00Z", "2023-12-31", "d"],
        ]
    )
    folder = tmp_path / "text" / "1"
    folder.mkdir(parents=True)
    for acc, words in [("k1", "alpha beta gamma " * 50), ("k2", "alpha beta delta " * 50)]:
        with gzip.open(folder / f"{acc}.txt.gz", "wt") as fh:
            fh.write(words)
    sims = compute_similarities(idx, tmp_path)
    assert len(sims) == 1 and 0 < sims["text_sim_cosine"].iloc[0] < 1
    dates = pd.bdate_range("2024-01-29", "2024-02-09", tz="UTC")
    frame = text_factor_frames(sims, dates)["text_sim_cosine"]["EQ:A"]
    assert np.isnan(frame["2024-02-01"])  # accepted that evening: not usable that day
    assert frame["2024-02-02"] == pytest.approx(sims["text_sim_cosine"].iloc[0])
