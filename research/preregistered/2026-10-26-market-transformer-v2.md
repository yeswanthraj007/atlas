# Pre-registration: market transformer v2 (price + news) and time-series forecasting

Committed while the v1 deep-pattern run is still in progress, before any v1 result is seen.
The v1 result may not change these rules.

## Question
1. Does a GPT-style transformer that reads each stock's daily price, volume and news as one
   sequence predict next-month relative returns better than the price-only v1 transformer?
2. Do pretrained time-series forecasting models (Chronos-Bolt) or our own transformer
   forecast volatility better than the HAR model already in the registry, and do they
   forecast returns at all?

## What "the same way as Anthropic" means here
Same architecture family and training recipe as large language models, scaled to our data:
- Decoder-only transformer with a causal mask (each day can only attend to earlier days),
  pre-norm layers, AdamW, warmup + cosine learning-rate schedule, gradient clipping.
- Stage 1, self-supervised pretraining (the analogue of next-token prediction): for every
  day in the sequence, predict the next day's return and next day's absolute return. Uses
  training-period data only.
- Stage 2, fine-tuning: the same network, with a new head, predicts whether the next 20-day
  return beats the cross-sectional median (same target as v1).
Disclosed limit: language models learn from trillions of tokens. We have about 4 million
stock-days and about 1 million news articles. Scale is the main thing we cannot copy.

## Data
S&P 500 daily bars and Alpaca news, 2016-01 to 2026-10.
One token per stock-day:
- returns and volume features as in v1;
- the mean all-mpnet-base-v2 embedding (pinned revision) of news published before that
  day's 16:00 ET close (after the close goes to the next day), projected to the model width;
- the log news count, and a "no news" flag.
Context: 120 days. Company names and tickers masked in the text, as in the news model.

## Splits (identical to v1)
- Train (pretraining and fine-tuning): labels known by 2022-12-31.
- Early stopping: 2023.
- Frozen test: 2024-01-01 onward. No retraining or tuning after test data is seen.

## Models
A. MarketGPT
- Width 128, 4 layers, 4 heads, dropout 0.1.
- Pretraining: up to 10 epochs; fine-tuning: up to 20 epochs; patience 3.
B. MarketGPT-price: the same model with the news inputs zeroed (ablation; isolates news).
C. Chronos-Bolt (Amazon, Apache-2.0, pinned revision), zero-shot.
- Inputs: 252 days of each stock's daily log returns and daily absolute returns.
- Forecasts: the next 20 days, from which we compute 20-day return and 20-day realised volatility forecasts.
D. Vol transformer: the MarketGPT backbone with a regression head for next-20-day log realised
   volatility.

## Success criteria
Return models (A, B, C-returns), each:
1. Test IC > 0 with t >= 2.0, using the monthly non-overlapping IC defined in v1.
2. Net top-minus-bottom decile mean > 0 at tau = 0.5, with liquidity-tier costs.
3. Test IC above the v1 ridge baseline.
News is said to add value only if A's test IC exceeds B's with a paired t >= 2.0 across months.

Volatility models (C-vol, D), each:
4. Lower out-of-sample QLIKE loss than HAR, with a Diebold-Mariano t >= 2.0 across months.
A volatility model that passes is registered as a candidate replacement for HAR.

## Trial accounting
Five trials (A, B, C-returns, C-vol, D) are logged in research/trials.jsonl. Every result is reported, pass or fail.
