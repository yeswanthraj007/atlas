# Pre-registration: frontier-LLM forward test on earnings releases v1

Committed before any code is written and before any prediction exists. A forward test
cannot be edited after predictions start, so everything that could be tuned is fixed here.

## Why a forward test
A frontier model trained through 2026 already knows how 2024 and 2025 turned out, so any
backtest on that period is contaminated (Glasserman & Lin; He, Lv, Manela & Wu; Gao, Jiang & Yan). The only clean
evidence is predictions made BEFORE the outcome, on events after the model's training data ends,
logged with timestamps nobody can alter afterwards. This is the experiment the language-model
reader result (it did not beat word counts) pointed to.

## Question
Does a frontier LLM, shown only a company-masked earnings release, produce probabilities that
beat (a) the base rate and (b) our word-count (TF-IDF) baseline, for
T1: the market-adjusted reaction of the first session after the release is positive;
T2: the market-adjusted return from that session's close to 20 trading days later is positive
    (the tradable part);
T3: the absolute reaction is above the median of that calendar quarter's S&P 1500 releases?
(Definitions as in the language-model-reader pre-registration.)

## Setup (fixed)
- Model: claude-sonnet-5-5 as the primary (the exact model id is stored with every prediction;
  any other model is a separate, labelled stream and never pooled). Temperature 0, one call per
  release, a single fixed prompt (the file src/atlas/llm_forward/prompt.py, version string
  PROMPT_V1; its SHA-256 is stored with every prediction). Output: JSON with p_reaction_up,
  p_drift_up, p_big_move (each 0-1) and a rationale of at most 60 words.
- Input: the press-release text, with the company's own name and ticker replaced by
  "[COMPANY]", the first 6,000 characters. Text only in v1 (no prices, no consensus, no
  statements); adding structured data would be a separate pre-registration.
- Universe: 8-K Item 2.02 releases of S&P 1500 companies filed on or after the start date.
- Eligibility, enforced by the harness at logging time: the prediction is recorded
  before the start of the reaction session (the first regular session whose close is at or
  after the filing: 09:30 ET the same day if filed before the open, the next trading day if
  filed after the close). Releases filed during market hours, or seen after the session began, are logged as
  ineligible and never scored.
- Training cutoff guard: only releases dated after the model's documented training-data end
  count (for claude-sonnet-5-5: after 2026-06-30, conservative). The prompt tells the model to ignore any outside knowledge of
  the company's later results.
- Tamper evidence: predictions go to an append-only file; the SHA-256 of each day's lines
  is written to research/forward/hashes/<date>.txt and committed and pushed to git, whose timestamp
  is the external record.
- Baselines logged alongside each prediction: the TF-IDF logistic and ridge models fitted on
  2014-2022 releases (frozen), and the base rates.
- Spend cap: stop the run when the month's estimated API cost exceeds $75.

## Metrics and decision rule (evaluated at fixed checkpoints only)
Checkpoints: when 600, 1,500 and 3,000 eligible predictions have a complete 20-day outcome. No
interim looks. Metrics: log loss, Brier score, AUC and accuracy for T1 and T3; for T2 the monthly
rank correlation (IC) between p_drift_up and the realised drift and the net top-minus-bottom decile
spread (liquidity-tier costs); a calibration table. Comparisons with the TF-IDF baseline use a paired
difference of per-release log loss with standard errors clustered by filing day.
Promotion (to paper-trading candidate): T2 IC > 0 with t >= 3, net spread > 0 and IC above the
baseline's; or T3 log loss below the TF-IDF baseline's with paired t >= 3. Anything else is a
rejection, reported in full.

## Trial accounting
One trial per checkpoint is logged (primary model, three targets together).
Disclosure: the same filings were used in earlier backtests with weaker models; this test
uses only releases that did not exist when the model was trained.

## Amendment 1 (2026-11-09, before any prediction was logged)
The API rejects the `temperature` parameter for claude-sonnet-5-5 ("Extra inputs are not
permitted"), so "temperature 0" in Setup cannot be applied. Calls use the model's default
sampling, one call per release, and the single logged output is the prediction (no re-draws,
no averaging). Nothing else changes. Found by the first real API call, on a synthetic snippet
that was not logged.
