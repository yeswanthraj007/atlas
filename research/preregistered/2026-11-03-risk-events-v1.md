# Pre-registration: risk and event predictors v1 (earnings-move size, drawdown)

Committed before any model code exists. Results may not change these rules.

## Why these targets
Direction of 20-day returns failed 65 trials (v3: 50.3% accuracy, IC 0.008). Move size and
downside risk are predictable (the volatility model: 76% above-median accuracy, t = 33).
The user's goal is 65%+ accuracy. That is only reachable where there is real structure, so
we test targets of that kind. Honest framing: for these targets, easy baselines may already
reach 65%. Accuracy alone is not value. Value is the margin over the naive baseline.

## Evidence
- Realised-volatility forecasting: simple HAR is hard to beat (Corsi 2009; "HARd to Beat",
  arXiv 2406.08041; Zhang et al., arXiv 2108.00480). Gains from ML are small.
- Crash/drawdown detection: ML reaches AUC about 0.74 vs 0.73 for HAR (arXiv 2604.17251).
- Attention and volatility: Wikipedia page views and Google search volume precede larger
  absolute moves (Moat et al. 2013; Preis, Moat & Stanley 2013).
- Earnings-announcement returns are partly predictable from uncertainty and from the
  firm's own history (Garfinkel et al.; Hunt, Myers & Myers).

## Data
S&P 1500 daily bars (adjusted), S&P 1500 membership rule as in the v3 pre-registration (survivorship
bias for S&P 400/600 disclosed), 8-K Item 2.02 earnings announcements 2014-2026
(data/sec/earnings_index.parquet), Wikipedia daily page views from 2015-07 for the 1,144
S&P 1500 companies that have an English article.

## Targets (all balanced by construction: the base rate is 50%)
**E1, earnings move size, cross-sectional.** An event is one 8-K Item 2.02 announcement
(first one per stock per day). Reaction = market-adjusted close-to-close return over the
first session that closes after the announcement time (close before the announcement to
close after it; equals news_day_index d0 vs d0-1). Target = 1 if |reaction| is above the
median |reaction| of all events in the same announcement calendar quarter.

**E2, earnings move size, versus the stock's own history.** Target = 1 if |reaction| is above
the median |reaction| of that stock's previous 8 events (needs at least 4). This removes
the "volatile stocks move a lot" level effect and asks whether this release is a surprise
in size.

**D1, drawdown risk.** Sample dates: every 21st trading day. Worst drawdown over the next 21
trading days = min over j = 1..21 of close(t + j) / close(t) - 1. Target = 1 if it is worse
than the cross-sectional median of that date.

## Features (all use data up to and including the last close before the outcome window)
Realised volatility (5, 21, 63, 252 days, log), 1-month and 3-month return, drawdown from the
252-day high, worst daily return of the last 21 days, return skew over 63 days, log
volume ratio (5 vs 60 days), log dollar volume (size proxy), market volatility (SPY, 21
days), days since the stock's last earnings announcement. Earnings tasks add: the stock's
last 1, 2 and 4 absolute reactions and its median of the last 8, announcement before or
after market hours (hour of the 8-K acceptance time), and number of prior events.
Wikipedia attention group (ablation): log of the 7-day average views / 90-day average, log
level of 90-day views, 7-day change.
Features are computed per date and per stock; no feature uses information after the
decision time.

## Models and splits
- Model: HistGradientBoostingClassifier. Hyperparameters chosen from {depth 3 or 5, 200
  or 400 iterations, learning rate 0.05} by 2023 log loss, trained on data with outcomes known by
  2022-12-31. The final model is refitted on outcomes known by 2023-12-31 with the chosen
  setting and then frozen. No tuning after the test window is opened.
- Test: all events or sample dates from 2024-01-01 to 2026-09-30 (outcomes known).
- Baselines (the strongest of them on the test is used, conservative for the model):
  - E1: rank of the stock's median |reaction| over its last 8 events; rank of 63-day vol.
  - E2: indicator "the last event's |reaction| was above that stock's median"; rank of
    5-day vol / 63-day vol.
  - D1: rank of 63-day vol; rank of 21-day vol; rank of a 3-term HAR forecast (OLS of the
    realised 21-day vol on lrv_5, lrv_21, lrv_63, fitted on the training period).
  Each baseline predicts "above" for stocks ranked in the upper half of its date (or
  quarter, for E1) and is scored the same way as the model.

## Metrics
Accuracy (primary), AUC, and accuracy of the model's most confident 20% on each side. All
scored by calendar month of the event or sample date. Paired monthly differences model minus best
baseline, with a t-statistic over months. Accuracy 95% intervals by month bootstrap.

## Success criteria (per target)
A. Target flag: test accuracy >= 65%.
B. Edge: test accuracy exceeds the best baseline's, paired monthly t >= 3.0 (raised from 2.0 to
   account for 65 earlier trials on this test window).
Promoted to the signal registry as a risk signal only if B holds and accuracy >= 60%.
If A holds but B does not, the result is reported as "reaches 65% but adds nothing over the
naive baseline". That is not promoted.

Wikipedia ablation, per target: same model without the attention group. Attention adds value
only if accuracy with it exceeds accuracy without it, paired monthly t >= 2.0.

## Trial accounting
Three trials (E1, E2, D1) plus three ablation trials are logged in research/trials.jsonl,
including the failed ones. Disclosure: the 2024-2026 window has been used by earlier
experiments on other targets, not these.
