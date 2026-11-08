# ruff: noqa: E501
import io
import zipfile

import numpy as np
import pandas as pd

from atlas.factors.insider import cluster_signal, qualifying_purchases, routine_purchase_mask
from atlas.marketdata.insider import dataset_links, parse_quarter


def _tx(rows):
    cols = [
        "owner_cik",
        "issuer_cik",
        "trans_date",
        "filing_date",
        "symbol",
        "code",
        "relationship",
        "value",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["trans_date"] = pd.to_datetime(df["trans_date"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def test_routine_means_same_month_in_each_of_the_three_previous_years():
    rows = [
        ("1", "9", f"{y}-03-15", f"{y}-03-17", "AAA", "S", "Director", 5e4)
        for y in (2019, 2020, 2021)
    ]
    rows.append(("1", "9", "2022-03-10", "2022-03-12", "AAA", "P", "Director", 5e4))  # routine
    rows.append(("2", "9", "2022-03-10", "2022-03-12", "AAA", "P", "Officer", 5e4))  # first-timer
    rows.append(("3", "9", "2021-03-10", "2021-03-12", "AAA", "P", "Officer", 5e4))
    rows.append(("3", "9", "2022-09-10", "2022-09-12", "AAA", "P", "Officer", 5e4))  # other month
    tx = _tx(rows)
    r = routine_purchase_mask(tx)
    assert bool(r.iloc[3]) and not bool(r.iloc[4]) and not bool(r.iloc[6])


def test_qualifying_purchases_filter_by_role_value_and_routine():
    tx = _tx([
        ("1", "9", "2022-05-01", "2022-05-03", "AAA", "P", "Director", 50_000),
        ("2", "9", "2022-05-01", "2022-05-03", "AAA", "P", "TenPercentOwner", 50_000),  # not a director/officer
        ("3", "9", "2022-05-01", "2022-05-03", "AAA", "P", "Officer", 5_000),  # too small
        ("4", "9", "2022-05-01", "2022-05-03", "AAA", "S", "Officer", 90_000),  # a sale
    ])  # fmt: skip
    q = qualifying_purchases(tx)
    assert list(q["owner_cik"]) == ["1"]


def test_cluster_signal_counts_distinct_insiders_with_a_one_day_lag():
    idx = pd.bdate_range("2022-05-01", periods=120, tz="UTC") + pd.Timedelta(hours=4)
    cols = pd.Index(["EQ:AAA", "EQ:BBB"])
    p = _tx([
        ("1", "9", "2022-05-02", "2022-05-03", "AAA", "P", "Director", 5e4),
        ("1", "9", "2022-05-10", "2022-05-11", "AAA", "P", "Director", 5e4),  # same insider again
        ("2", "9", "2022-05-20", "2022-05-23", "AAA", "P", "Officer", 5e4),
    ])  # fmt: skip
    s = cluster_signal(p, idx, cols, window=60)
    day = lambda d: s.index.get_indexer([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4)])[0]  # noqa: E731
    assert s["EQ:AAA"].iloc[day("2022-05-03")] == 0  # filed that day: usable only from the next day
    assert s["EQ:AAA"].iloc[day("2022-05-04")] == 1
    assert s["EQ:AAA"].iloc[day("2022-05-12")] == 1  # one insider, counted once
    assert s["EQ:AAA"].iloc[day("2022-05-24")] == 2
    assert (
        s["EQ:AAA"].iloc[day("2022-07-12")] == 1
    )  # insider 1's windows have expired, 2's have not
    assert s["EQ:BBB"].sum() == 0
    # the signal on day t does not depend on filings made on or after t
    later = _tx([("1", "9", "2022-06-01", "2022-06-02", "AAA", "P", "Director", 5e4)])
    s2 = cluster_signal(pd.concat([p, later]), idx, cols, window=60)
    np.testing.assert_array_equal(
        s["EQ:AAA"].iloc[: day("2022-06-03")], s2["EQ:AAA"].iloc[: day("2022-06-03")]
    )


def test_parse_quarter_reads_the_sec_tables():
    def tsv(df):
        return df.to_csv(sep="\t", index=False)

    sub = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a2"], "FILING_DATE": ["05-JAN-2024", "06-JAN-2024"],
                        "DOCUMENT_TYPE": ["4", "4/A"], "ISSUERCIK": ["1", "1"],
                        "ISSUERTRADINGSYMBOL": ["brk-b", "brk-b"], "AFF10B5ONE": ["0", "0"]})  # fmt: skip
    tr = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a1", "a2"], "TRANS_DATE": ["03-JAN-2024"] * 3,
                       "TRANS_CODE": ["P", "A", "P"], "TRANS_SHARES": ["100", "5", "9"],
                       "TRANS_PRICEPERSHARE": ["50", "0", "1"], "TRANS_ACQUIRED_DISP_CD": ["A"] * 3})  # fmt: skip
    own = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a2"], "RPTOWNERCIK": ["7", "7"],
                        "RPTOWNERNAME": ["X"] * 2, "RPTOWNER_RELATIONSHIP": ["Director", "Director"],
                        "RPTOWNER_TITLE": [None, None]})  # fmt: skip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", tsv(sub))
        z.writestr("NONDERIV_TRANS.tsv", tsv(tr))
        z.writestr("REPORTINGOWNER.tsv", tsv(own))
    out = parse_quarter(buf.getvalue())
    assert len(out) == 1  # amendment and non-P/S code dropped
    r = out.iloc[0]
    assert (
        r["symbol"] == "BRK.B"
        and r["value"] == 5000.0
        and r["filing_date"] == pd.Timestamp("2024-01-05")
    )


def test_dataset_links_picks_quarters_and_keeps_every_path():
    html = ('<a href="/files/structureddata/data/insider-transactions-data-sets/2016q1_form345.zip">x</a>'
            '<a href="/files/datastandardsinnovation/data/insider-transactions-data-sets/2026q2_form345.zip">y</a>'
            '<a href="/files/structureddata/data/insider-transactions-data-sets/2010q1_form345.zip">z</a>')  # fmt: skip
    links = dataset_links(html)
    assert list(links) == ["2016q1", "2026q2"]


def test_parse_quarter_tolerates_older_files_without_newer_columns():
    sub = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1"],
            "FILING_DATE": ["05-JAN-2016"],
            "DOCUMENT_TYPE": ["4"],
            "ISSUERCIK": ["1"],
            "ISSUERTRADINGSYMBOL": ["AAA"],
        }
    )  # no AFF10B5ONE column
    tr = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1"],
            "TRANS_DATE": ["04-JAN-2016"],
            "TRANS_CODE": ["P"],
            "TRANS_SHARES": ["10"],
            "TRANS_PRICEPERSHARE": ["20"],
            "TRANS_ACQUIRED_DISP_CD": ["A"],
        }
    )
    own = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1"],
            "RPTOWNERCIK": ["7"],
            "RPTOWNERNAME": ["X"],
            "RPTOWNER_RELATIONSHIP": ["Officer"],
            "RPTOWNER_TITLE": ["CEO"],
        }
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, df in (
            ("SUBMISSION.tsv", sub),
            ("NONDERIV_TRANS.tsv", tr),
            ("REPORTINGOWNER.tsv", own),
        ):
            z.writestr(name, df.to_csv(sep="\t", index=False))
    out = parse_quarter(buf.getvalue())
    assert len(out) == 1 and not bool(out.iloc[0]["aff10b5one"]) and out.iloc[0]["value"] == 200.0
