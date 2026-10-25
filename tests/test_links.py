import pandas as pd

from atlas.factors.links import company_aliases, extract_links, link_matrix, name_finder

CONS = pd.DataFrame(
    {
        "Symbol": ["AAPL", "INTC", "TGT", "BALL"],
        "Security": ["Apple Inc.", "Intel Corporation", "Target Corporation", "Ball Corporation"],
    }
)


def test_aliases_drop_suffixes_but_keep_ambiguous_names_full() -> None:
    a = company_aliases(CONS)
    assert "Apple" in a["AAPL"] and "Intel" in a["INTC"]
    assert a["TGT"] == ["Target Corporation"]  # "Target" alone is an ordinary word
    assert a["BALL"] == ["Ball Corporation"]


def test_extracts_typed_links_and_ignores_lowercase_words() -> None:
    finder = name_finder(company_aliases(CONS))
    text = (
        "Apple accounted for 18% of our net revenue in fiscal 2025. "
        "We compete with Intel in data center processors. "
        "Our target market grew. We hit the ball out of the park."
    )
    links = extract_links(text, "QCOM", "2025-11-05T21:00:00Z", finder)
    kinds = {(lk.target, lk.kind) for lk in links}
    assert ("AAPL", "customer") in kinds and ("INTC", "competitor") in kinds
    assert not any(lk.target in {"TGT", "BALL"} for lk in links)
    assert not extract_links("Intel builds chips.", "INTC", "2025-01-01", finder)  # no self


def test_link_matrix_uses_latest_filing_before_date() -> None:
    links = pd.DataFrame(
        {
            "source": ["QCOM"] * 4,
            "target": ["AAPL", "AAPL", "INTC", "INTC"],
            "kind": ["customer"] * 4,
            "sentence": ["s"] * 4,
            "accepted": pd.to_datetime(
                ["2024-11-01", "2024-11-01", "2025-11-01", "2025-11-01"], utc=True
            ),
        }
    )
    m = link_matrix(links, pd.Timestamp("2025-06-01", tz="UTC"), "customer")
    assert dict(m["QCOM"]) == {"AAPL": 2}  # the 2025 filing is not public yet
    m2 = link_matrix(links, pd.Timestamp("2026-01-01", tz="UTC"), "customer")
    assert dict(m2["QCOM"]) == {"INTC": 2}
