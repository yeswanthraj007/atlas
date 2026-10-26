# Pre-registration: deep price-pattern models v1

Committed before the model code exists. Results may not change these rules.

## Question
Can deep networks that read raw price and volume history (a chart-image CNN and a patch
transformer) predict next-month relative returns out of sample, after costs, better than
our simple price signals?

## Evidence
- Jiang, Kelly & Xiu (2023, J. Finance), "(Re-)Imag(in)ing Price Trends": a CNN on images
  of OHLC-volume charts predicts future returns out of sample.
- Patch transformers (PatchTST, Nie et al. 2023) and time-series foundation models
  (Chronos, TimesFM) forecast competitively, but recent finance evaluations find weak
  trading value; hence the requirement to beat a simple baseline.

## Data
S&P 1500 daily bars (adjusted), 2016-01 to 2026-10. Sample dates every 5 trading days.
Survivorship bias for S&P 400/600 names disclosed (today's members).

## Models
A. ChartCNN: 20-day chart image, 64 x 60 pixels (3 rows of pixels per day: open tick,
   high-low bar, close tick; 20-day moving-average line; volume bars in the bottom fifth).
   3 conv blocks (5x3 kernels, 64/128/256 channels, batch norm, leaky ReLU, 2x1 max pool),
   dropout 0.5, linear head. Target: is the next 20-day return above the cross-sectional
   median? (binary, cross-entropy).
B. PatchTransformer: last 120 days of daily log returns and log volume ratio, normalised
   per sample; patches of 10 days -> 12 tokens; 3 encoder layers, 4 heads, width 64;
   same target.
Training: samples with outcome known by 2022-12-31; early stopping on 2023 samples;
AdamW, lr 1e-3 (CNN) / 3e-4 (transformer), batch 512, at most 20 epochs, patience 3.
Model is then frozen. No retraining, no tuning after 2024 data is seen.

## Baseline
Ridge on the same samples' simple features: 1-month reversal, 12-1 momentum, 1-year
volatility (cross-sectional ranks), trained and frozen identically.

## Test (frozen): 2024-01-01 to the last date with a known 20-day outcome
Primary metric: monthly (non-overlapping, every 4th sample date) cross-sectional IC of
the model's up-probability vs realised 20-day returns.
Also: direction accuracy, top-minus-bottom decile monthly return gross and net of
liquidity-tier costs (tau = 1 and 0.5), by segment (S&P 500 vs mid/small), per ticker.

## Success criteria (per model)
1. Test IC > 0 with t >= 2.0.
2. Net-of-cost top-minus-bottom decile mean > 0 (tau = 0.5).
3. Test IC exceeds the ridge baseline's test IC.
Passing all three registers the model as a candidate direction signal for paper trading.

## Trial accounting
Two trials (one per architecture) plus one baseline trial, logged in research/trials.jsonl.
