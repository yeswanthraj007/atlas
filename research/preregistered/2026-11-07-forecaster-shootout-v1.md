# Pre-registration: forecaster shootout v1 (5 models, 2 tasks)

Committed before any shootout code or result exists. Results may not change these rules.

## Question
Which family of time-series forecaster is best for stocks: a classical model (ARIMA-GARCH), a
recurrent network (LSTM), our own transformer (MarketGPT), or a pretrained foundation model
(Google TimesFM, Amazon Chronos)? Ranked on two tasks, with simple baselines for scale.

## Models
1. ARIMA-GARCH: ARMA(1,1) mean and GARCH(1,1) variance, fitted per stock (arch package),
   parameters estimated once on data to 2023-12-31, then frozen. The 21-day variance forecast
   is the closed-form GARCH sum; the return forecast is the ARMA 21-day sum.
2. LSTM: two layers, 128 hidden units, same 120-day inputs as MarketGPT with the market-return
   input hidden. One model per task: rank loss for returns, MSE on log volatility for
   volatility. Trained on outcomes known by 2022-12-31, early-stopped on 2023, frozen.
3. MarketGPT: our existing frozen transformer. Returns: the v3 price-only models
   (price_A/B/C seed 0, walk-forward). Volatility: the v2 volatility transformer. Used as is,
   not retrained.
4. Chronos-Bolt-small (Amazon, Apache-2.0, pinned revision), zero-shot.
5. TimesFM 2.5 200M (Google, Apache-2.0, pinned revision), zero-shot.
Baselines: ridge on rank features (reversal, 12-1 momentum, 1-year volatility) for returns;
HAR (OLS on log realised vol over 5, 21 and 63 days, fitted to 2022) and "naive" (last
21-day volatility) for volatility.

Inputs for the two foundation models: returns task, the last 252 daily log returns. Volatility
task, the series of log 5-day realised volatility (best practice for them; fixes a flaw in
the earlier Chronos volatility test, which fed absolute returns), forecast 21 steps ahead.
Zero-shot: no fine-tuning, no tuning of any kind on our data.

## Data and test
S&P 500 members (membership rule as in earlier pre-registrations). Test dates: every 21st
trading day from 2024-01-02 until the last date with a known 21-day outcome.
Frozen: no choice is made after the first test date is scored.

## Tasks and metrics
R, 21-day return ranking: mean cross-sectional Spearman IC between the forecast and the
realised 21-day return, over test dates; t-statistic; plus beats-median accuracy.
V, 21-day volatility: QLIKE (Patton 2011) of the volatility forecast against realised
21-day volatility; Diebold-Mariano t of each model against HAR; plus above-median accuracy.

## Ranking
Rank the five models on each task (R by IC, V by QLIKE; ties by the other metric). Overall rank
= the mean of the two ranks. "Best" overall = lowest mean rank. A model is called useful
for a task only if it beats the baseline for that task with t >= 3.0 (stricter than 2 to
account for 69 earlier trials): ridge for R, HAR for V.
Whatever the ranking, the report states plainly which models are useful and which are
not.

## Trial accounting
Five trials (one per model) are logged in research/trials.jsonl with both tasks' metrics.
Disclosure: the 2024-2026 window was used by earlier experiments on related targets
(v2 and v3 forecasters, Chronos). Chronos and MarketGPT results here are new evaluations
on new dates and a new input for Chronos volatility.
