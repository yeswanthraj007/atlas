# Pre-registration: market intraday momentum on liquid ETFs (real-time candidate)

Committed before any minute data is downloaded or any code written. Results may not change
these rules.

## Why
Every daily and monthly direction test failed. Information at shorter horizons is richer, there are
many more independent observations (about 250 a year instead of 12), and a liquid ETF has
tiny costs. This is also the natural first target for a real-time bot.

## Evidence
Gao, Han, Li & Zhou (2018, J. Financial Economics), "Market Intraday Momentum": on the S&P 500
ETF, 1993-2013, the first half-hour return (from the previous close) predicts the last
half-hour return, and the effect is stronger on volatile days, high-volume days and
macro-news days. Our data (2016-2026) is entirely after their sample, so it is genuinely
out-of-sample for the published finding. Whether it persisted, or was arbitraged away after
publication (McLean & Pontiff 2016), is the open question. Several follow-ups report it in
other markets.

## Data
Alpaca 1-minute bars (SIP, adjusted for splits and dividends) for SPY (primary), QQQ and IWM
(robustness), 2016-01-01 to the last complete session, regular hours 09:30-15:59 ET.
Days that lack any needed bar (early closes, outages) are dropped, and the count is reported.

## Definitions (all from bars already complete at the decision time)
- r1 = ln(price at 10:00 / previous day's close): the close of the 09:59 bar over the close of
  the previous 15:59 bar.
- r12 = ln(price at 15:30 / price at 15:00): the close of the 15:29 bar over the close of the 14:59 bar.
- Decision at 15:30; entry at the OPEN of the 15:30 bar; exit at the CLOSE of the 15:59 bar.
- Trade return = ln(close 15:59 / open 15:30).
- vol20 = standard deviation of the previous 20 daily close-to-close returns, known at the start of the day.

## Strategies (fixed now)
A (primary): long if r1 > 0, short otherwise, for the last half-hour.
B (secondary): the sign of the predicted last-half-hour return from an OLS regression of
   ln(close 15:59 / close 15:29) on r1 and r12, fitted on the development period.
C (secondary, the volatility interaction): strategy A only on days when vol20 is in the top
   third, with the cut-off (the 67th percentile) fixed on the development period; flat
   otherwise.
Cost: 1.0 bp round trip on the primary run (SPY's spread is about 0.2 bp; the rest is
slippage). Sensitivity: 0.5 and 2.0 bp.

## Splits
Development 2016-01-01 to 2022-12-31: only the regression coefficients (B) and the volatility cut-off
(C) are estimated here. Nothing else is tuned.
Frozen test: 2023-01-01 to the last full day. No change after the test is opened.

## Metrics
Per trade: net return in bp; hit rate; t-statistic of the mean net return (over days); annualised
Sharpe of daily P&L (flat days count as zero); worst drawdown. Reported for SPY, QQQ and IWM,
and by calendar year. Same on the development period for context.

## Success criteria (strategy A on SPY, frozen test)
1. Mean net return per trade > 0 with t >= 3.0 (raised from 2.0 to account for 80+ earlier trials).
2. Hit rate above 51%.
3. Net Sharpe >= 0.5.
All three: promoted to paper trading as an execution candidate (not to real money).
Strategies B and C are reported either way. C counts as promising only if its mean net
return per trade exceeds A's by a paired t >= 2 on the frozen test.

## Trial accounting
Three trials (A, B, C) are logged in research/trials.jsonl; QQQ and IWM are robustness
rows inside A's metrics. Disclosure: no intraday result has been seen before this was written.
