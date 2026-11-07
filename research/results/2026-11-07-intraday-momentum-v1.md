# Result: market intraday momentum on SPY, QQQ, IWM — rejected

Pre-registration: research/preregistered/2026-11-07-intraday-momentum-v1.md. Trials #81 (A),
#82 (B), #83 (C). 1-minute SIP bars, 2016-2026; development 2016-2022 (1,750 SPY sessions),
frozen test 2023 to October 2026 (962 SPY sessions). Cost 1.0 bp round trip.

## SPY, frozen test
| Strategy | Trades | Net bp/trade | t | Hit rate | Sharpe |
|---|---|---|---|---|---|
| A: long if the first half-hour is up | 959 | -0.63 | -0.89 | 47.1% | -0.45 |
| B: regression-signed | 962 | -1.72 | -2.44 | 45.3% | -1.25 |
| C: A on high-volatility days only | 122 | -4.34 | -1.58 | 44.3% | -0.80 |

Criteria for A: net > 0 with t >= 3 FAIL; hit rate > 51% FAIL; Sharpe >= 0.5 FAIL. Rejected.
Cost sensitivity for A: 0.5 bp -0.13 bp/trade (t -0.18); 2.0 bp -1.63 bp/trade (t -2.31).
By year, A: 2023 +0.97 bp, 2024 -1.28, 2025 -1.04, 2026 -1.24 (none significant).
QQQ and IWM, frozen test, A: -1.38 bp (t -1.59) and -0.87 bp (t -1.08).

## Development period (2016-2022)
Strategy A lost on SPY (-1.70 bp/trade, t -2.04, hit rate 45.4%) and on IWM (-2.52, t -2.86). The sign
was the OPPOSITE of the published pattern: after 2016 the first half-hour tended to reverse by
the close. A reversal rule would earn about +0.7 bp gross, which 1 bp of cost removes.

## Volatility interaction (pre-registered secondary)
Published: the effect is stronger on volatile days. Here, A on the top-third volatility
days lost 4.34 bp/trade against -0.09 bp on other days (Welch t -1.50): the opposite sign,
not significant.

## Reading
The published 1993-2013 effect did not persist: it has no profitable form at SPY's costs
after 2016. This fits the post-publication decay of anomalies (McLean & Pontiff 2016). Strategy
interpretation note: "C beats A" was compared as high-volatility trades against the remaining
days (Welch t), because on the days C trades it equals A by construction.
