# Pre-registration: MarketGPT v3 (more data, better training)

Written after the v2 results and the engineering diagnostics. Committed before the v3 code
exists. Results may not change these rules.

## Why v3
Diagnostics (scripts/diagnose_market_gpt.py, data up to 2023 only):
- Planted-signal test: MarketGPT recovered 90% of a planted pattern (2023 IC +0.230 of a
  +0.256 ceiling). The pipeline works.
- The saved v2 news model learned something weak and real that decays over time:
  training-years IC +0.079, 2023 IC +0.049, test 2024-26 +0.030. News beat price-only in
  every period.
The diagnosis is too little data for a weak signal, and a training target that does not
match how the model is used. v3 changes data and training, not the architecture.

## Disclosure
The 2024-2026 test period was already used for v2, and v3's design is informed by v2
(e.g. keeping news). The success bar is raised to t >= 3.0 to account for 63 earlier trials
(Harvey, Liu & Zhu 2016). From 2026-10-27 the frozen ensemble is also scored on new data
going forward; this is the only fully unseen test.

## Data (changes from v2)
- Universe: S&P 1500 (v2: S&P 500), about 3x more stocks.
  - S&P 500 membership comes from index-addition dates.
  - Mid/small caps count from their first price. Survivorship bias, disclosed.
- Fine-tuning samples: every trading day (v2: every 5th), about 5x more.
- News: existing Alpaca news; 861 of 1,000 mid/small names have 50 or more articles.
- Pretraining: windows every 5th day; every token is a target, so all days are covered.
- Features, causal masking, identity masking and 64-dimension news PCA are unchanged from v2.

## Training
- Backbone: MarketGPT, width 128, 4 layers, 4 heads.
  - Width 256 was not chosen: the diagnostics show mild overfitting, not underfitting, and
    it trains 2x slower.
- Stage 1, pretraining: next-day return and absolute return, on data up to 2022-12-31.
  Early-stopped on 2023. At most 8 epochs of all windows, patience 2.
- Stage 2, fine-tuning: one score per stock-day.
  - Batches are 8 dates x 64 stocks.
  - Each epoch draws 400,000 samples. At most 8 epochs, patience 2.
  - Early-stopped on the mean 2023 cross-sectional IC.
- Choice of loss, made on 2023 IC only, with the refit A split, seed 0:
  - (a) listwise ranking loss: 1 - Pearson correlation between scores and Gaussian-ranked
    20-day returns within each date;
  - (b) the binary beats-median cross-entropy used in v2.

  The better loss is then tried at fine-tuning learning rate 1e-4 against 3e-4, still on
  2023 IC only. These are the only tuning choices.
- Ensemble: 3 seeds of the chosen setting. Scores are z-scored within each date and averaged.

## Walk-forward retraining (annual)
Each refit is fine-tuned from the same backbone, pretrained on data up to 2022.

| Refit | Trains on labels known by | Early-stopped on | Predicts |
|---|---|---|---|
| A | 2022-12-31 | 2023 | 2024 |
| B | 2023-12-31 | 2024 | 2025 |
| C | 2024-12-31 | 2025 | 2026 |

## News ablation
The same pipeline with news inputs zeroed: its own pretraining, then refits A, B and C, seed 0
only. It is compared with the news model's seed 0.

## Baseline
Ridge (alpha 10) on 1-month reversal, 12-1 momentum and 1-year volatility ranks.
It is refitted with the same A, B and C splits.

## Test
Monthly test dates are every 20th trading day from 2024-01-02, so 20-day outcomes do not
overlap. The test runs to the last date with a known outcome.

Metrics:
- monthly cross-sectional IC and its t-statistic;
- top-minus-bottom decile spread, gross and net of liquidity-tier costs, at tau 1 and 0.5;
- results for the S&P 500 and mid/small segments;
- per-ticker results for 10 names.

## Success criteria (ensemble)
1. Test IC > 0 with t >= 3.0.
2. Net top-minus-bottom decile mean > 0 at tau 0.5.
3. Test IC above the ridge baseline's.

Outcomes:
- All three pass: registered as a candidate direction signal for paper trading.
- t between 2.0 and 3.0, with criteria 2 and 3 passing: forward paper-tracking only.
  It may not rank picks.
- Otherwise: rejected.

News adds value only if seed-0 news IC beats seed-0 price-only IC with a paired monthly
t >= 2.0.

## Trial accounting
Two trials are logged in research/trials.jsonl: the ensemble and the price-only ablation.
The baseline is logged as a comparison.

## Amendment 1 (2026-10-26, before any v3 result; the run was stopped while still embedding news)
Made at the user's request, so that v3 does everything the plan promised:
1. Mid/small-cap news. A separate Alpaca news download for the 1,000 S&P 400/600 names goes
   to data/news_midsmall. It is merged with the existing news and de-duplicated by article id.
2. Model size is chosen on 2023, not fixed:
   - The width-128 backbone first picks the loss (rank vs classify, lr 3e-4).
   - A width-256 backbone (its own pretraining) is then compared at lr 3e-4.
   - The better width is tried at lr 1e-4.
   - Every choice uses refit A, seed 0, and 2023 IC only.
   - The price-only ablation uses the chosen width.
Nothing else changes: the success criteria, test, baseline and trial accounting are as above.

## Amendment 2 (2026-10-26; based on 2023 validation only; no 2024+ metric has been computed)
Problem found: in the first full run, fine-tuning memorised history. Training correlation
rose to 0.38 while 2023 IC stayed negative (best -0.019). Cause: the market-return input is
identical for every stock on a date, so its 120-day history fingerprints the date. The model
learned which kinds of stocks did well in each past period.
Exploration on 2023 only (scripts/explore_v3_fixes.py, refit A, width 128):

| Variant | Best 2023 IC | Training correlation |
|---|---|---|
| No fix | -0.019 | 0.38 |
| Market input hidden (3 seeds) | +0.071, +0.021, +0.053 (mean +0.048) | about 0.09 |
| Less overlap (stride 5), market input kept | -0.032 | 0.23 |
| Hidden + stride 5 | -0.014 | 0.12 |
| Hidden + stride 5 + dropout 0.3, weight decay 0.2 | -0.003 | 0.10 |

Change: every fine-tuning run, and every prediction, hides the market-return input
(`drop_market=True`). Pretraining is unchanged; next-day market context is legitimate there.
Overlap and regularisation are unchanged (neither helped). The tuning run made under the old
setup is archived, not reused. The loss, width and learning-rate choices are re-made under
the new setup, as in Amendment 1. Success criteria, test, baseline and trial accounting are
unchanged.

## Amendment 3 (2026-10-26 23:30, before the frozen test is opened): descriptive metrics
Added at the user's request so that "accuracy" is reported in the forms that matter for a
portfolio. These are reported only. They do NOT change the success criteria or the verdict.
They are computed by a separate script from the saved, frozen ensemble on the same monthly
test dates:
1. Monthly win rate: the share of test months in which the net top-minus-bottom decile
   spread (tau 0.5) is positive, with a binomial p-value against 50%.
2. Confident-call accuracy: beats-median accuracy for the model's top and bottom 5% and 10%
   of each month, compared with all stocks.
3. Long-only check: the share of months in which the top decile beats the equal-weight
   S&P 1500 average.
