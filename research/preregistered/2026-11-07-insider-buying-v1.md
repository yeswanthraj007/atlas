# Pre-registration: opportunistic insider cluster buying (direction signal)

Committed before any insider data is downloaded or any code is written. Results may not
change these rules.

## Why this, and why now
Eighty-plus trials on price, volume, news, filing text, links and foundation models gave no
direction signal that survives a frozen test (best: IC about 0.01-0.02, accuracy about 50.5%).
More architectures on the same information do not help. This tests NEW information: what
a company's own officers and directors do with their own money, reported to the SEC within two
business days and public. It is free, with history back to 2006.

## Evidence
- Lakonishok & Lee (2001, RFS): insider purchases predict returns, mostly in smaller firms.
- Cohen, Malloy & Pomorski (2012, J. Finance), "Decoding Inside Information": "routine"
  insider trades (same calendar month for at least three years) carry no information; "opportunistic"
  trades earn large risk-adjusted returns (about 5% over six months in their sample).
- Clustered purchases (two or more insiders in a short window) are followed by larger
  abnormal returns than solitary ones (industry and academic summaries of the same data).
Caveats: the effect is concentrated in smaller, less-followed firms; much of it may have been
arbitraged away since 2012; open-market purchases are rarer than sales (most stock-days have
no signal).

## Data
SEC "Insider Transactions Data Sets" (Forms 3, 4 and 5, parsed quarterly files) 2013Q1 to the
latest quarter. 2013-2015 is used only to classify routine trading in 2016. Universe: S&P 1500
(membership rule as before), matched by the issuer's trading symbol.
Timing: a filing is usable at the close of the trading day AFTER its filing date (filings
can arrive after the close), so the signal on day t uses filings dated up to t-1.

## Signal (one primary variant, fixed now)
Events: Form 4 non-derivative transactions with code P (open-market purchase), acquired
(A), by a reporting owner who is an officer or director, value (shares x price) of at least
$10,000.
Routine rule (simplification of Cohen-Malloy-Pomorski, disclosed): an insider-company pair
is "routine" for a purchase in month m of year Y if that insider made an open-market purchase or
sale (codes P or S) in month m in each of the years Y-1, Y-2 and Y-3. Routine purchases are dropped.
Everything else counts, including first-time buyers (a deliberate departure: CMP drop insiders
with too little history).
Signal on day t = the number of distinct non-routine insiders with qualifying purchases filed in
the 60 calendar days up to and including t-1; 0 when there are none.

## Test: the research lab's fixed gates (docs/RESEARCH_PROCESS.md)
Run `atlas research run insider_cluster_buying --universe sp1500`: look-ahead check, then
development 2017-2024 (IC t >= 3 and the same sign in at least 60% of years), then the
frozen holdout 2025 on (same sign, t >= 2), then deflated Sharpe across all logged trials.
The 21-day horizon is the lab's standard. Frozen holdout is opened only if development passes.

## Secondary report (not a gate)
Top-minus-bottom after liquidity-tier costs; results for S&P 500 vs mid/small caps; the
average abnormal return in the 21 days after the first qualifying purchase at each stock.

## Trial accounting
One trial (the primary variant) is logged. Any later variant (different window, value
thresholds, adding sales) is a new trial under a new name with its own pre-registration.
