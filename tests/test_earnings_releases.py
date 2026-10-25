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


def test_downloads_only_the_exhibit_via_the_index_page(tmp_path) -> None:
    import httpx

    from atlas.marketdata.earnings_releases import (
        EarningsRelease,
        EarningsReleaseDownloader,
        exhibit_url,
        read_release,
    )
    from atlas.marketdata.filings import FilingClient

    index = (
        '<tr><td><a href="/Archives/edgar/data/7/0001/ex991.htm">ex991.htm</a></td>'
        '<td>EX-99.1</td></tr><tr><td><a href="/Archives/edgar/data/7/0001/g1.jpg">'
        "g1.jpg</a></td><td>GRAPHIC</td></tr>"
    )
    assert exhibit_url(index) == "https://www.sec.gov/Archives/edgar/data/7/0001/ex991.htm"
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path.endswith("-index.htm"):
            return httpx.Response(200, content=index.encode())
        return httpx.Response(200, content=b"<p>" + b"Acme reports record revenue. " * 20 + b"</p>")

    client = FilingClient(
        "T t@example.com",
        tmp_path,
        min_interval=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    dl = EarningsReleaseDownloader(client, tmp_path)
    rel = EarningsRelease(7, "ACME", "0001", "2025-01-30", "2025-01-30T21:00:00Z", "2.02")
    assert dl.fetch(rel) is None
    assert not any(p.endswith(".txt") for p in seen)  # never pulled the big submission file
    assert read_release(tmp_path, 7, "0001").startswith("Acme reports record revenue")
