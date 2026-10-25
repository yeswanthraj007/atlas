from atlas.marketdata.earnings_releases import extract_press_release, list_releases

SUBMISSION = """<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>8-K
<TEXT><html><p>Item 2.02 Results of Operations. See Exhibit 99.1.</p></html></TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<TEXT><html><p>Acme reports record revenue of $5.1 billion, up 12% year over year.</p>
<p>Raises full-year guidance.</p></html></TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>GRAPHIC
<TEXT>begin 644 logo.jpg</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>"""


def test_extracts_the_press_release_exhibit() -> None:
    text = extract_press_release(SUBMISSION)
    assert text.startswith("Acme reports record revenue") and "Raises full-year" in text
    assert "Item 2.02" not in text and "logo" not in text
    only_8k = SUBMISSION.replace("EX-99.1", "EX-101.SCH")
    assert "Item 2.02" in extract_press_release(only_8k)  # fallback to the 8-K body


def test_lists_only_item_202_eight_ks_since_a_date() -> None:
    subs = {
        "cik": "7",
        "filings": {
            "recent": {
                "form": ["8-K", "8-K", "10-Q", "8-K"],
                "accessionNumber": ["a1", "a2", "a3", "a4"],
                "filingDate": ["2025-01-30", "2025-02-15", "2025-05-01", "2013-01-30"],
                "acceptanceDateTime": ["2025-01-30T21:05:00Z"] * 4,
                "items": ["2.02,9.01", "5.02", "", "2.02"],
            }
        },
    }
    rels = list_releases(subs, "ACME", since="2014-01-01")
    assert [r.accession for r in rels] == ["a1"]
    assert rels[0].symbol == "ACME" and rels[0].cik == 7
