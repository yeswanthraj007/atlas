# Pre-registration: language-model reader of earnings releases v1

Committed before any model is trained. Results may not change these rules.
Background: reports/LLM return prediction: research synthesis.md.

## Question
Can a fine-tuned language model that READS an earnings press release (and nothing else)
predict (T1) which way the stock reacts, (T2) the post-reaction drift over the next 20 days
(the tradable part), and (T3) whether the reaction will be larger than the quarter's
median, better than a bag-of-words baseline? T3 is where our one validated edge lives (E2,
E1), so it is the place an LLM could add most.

## Model and why this one
FacebookAI/roberta-base (125M parameters, MIT), pinned revision
e2da8e2f811d1448a5b465c236feacd80ffbac7b. Its training text ends in 2019, so it cannot know
anything about the 2024-2026 test window: the chronological-consistency property the literature
treats as essential (Glasserman & Lin; He, Lv, Manela & Wu; Gao, Jiang & Yan). It is a "language
model" in the sense of Chen, Kelly & Xiu (BERT/RoBERTa/OPT), not a chat-scale model; we chose
it because it runs well on the Apple GPU (DeBERTa-v3 hung on it) and cleanly predates the test.
Fine-tuned end to end with three output heads on the first token's representation.

## Data
The 8-K Item 2.02 earnings releases for the S&P 1500 (61,048 downloaded, 2014-2026), with the
membership rule used in earlier pre-registrations. Event timing and the reaction session
as in the risk-events pre-registration (an after-hours release is measured on the next
session). Input: the press-release text after the exhibit header, with the company's own name and
ticker replaced by "[COMPANY]", the first 384 tokens, always padded to exactly 384
(fixed shape; changing shapes is very slow on this GPU).

## Targets (all use only future prices; the model never sees prices)
- T1: 1 if the market-adjusted return of the reaction session is positive.
- T2: the Gaussian-ranked (within the announcement quarter) market-adjusted return from the
  close of the reaction session to the close 20 trading days later. Regression.
- T3: 1 if the absolute reaction is above the median absolute reaction of the announcement
  quarter (the same as target E1).
Loss: BCE(T1) + BCE(T3) + MSE(T2), equal weights, fixed.

## Splits
Train: events whose T2 outcome was known by 2022-12-31. Validation: events announced in
2023. Test (frozen): events from 2024-01-01 to 2026-09-30 with known outcomes.
AdamW, learning rate 2e-5, weight decay 0.01, batch 16, linear warm-up over 6% of steps then linear decay,
at most 3 epochs, seed 0. The epoch with the lowest validation total loss is kept. Then frozen.

## Baselines (same text, same splits, trained on the training split only)
TF-IDF (word 1-2-grams, 100k features, sublinear tf) + logistic regression (T1, T3) and
ridge (T2). For T3 the earlier price-feature gradient-boosting result (E1: 63.4% accuracy on
a similar test set) is quoted for context only.

## Metrics
T1 and T3: accuracy and AUC. T2: the rank correlation (IC) between the predicted score and the
realised drift, computed per announcement month (at least 30 events), its mean and t-statistic
across months; top-minus-bottom decile mean drift, gross and net of the liquidity-tier cost.
Paired monthly differences against the baselines. T1 is a "nowcast" of the immediate
reaction, not tradable by us (the price moves within seconds of the release), and is
reported only.

## Success criteria
Primary (tradable): T2 test IC > 0 with t >= 3.0 by month, and net decile spread > 0, and the
IC exceeds the TF-IDF baseline's.
Secondary: T3 accuracy >= 65%, above the TF-IDF baseline by a paired monthly t >= 2.0.
Passing the primary criterion registers the model as a candidate direction signal for paper
trading. T1 and T3 are never promoted by themselves.

## Trial accounting and disclosure
Three LLM trials (T1, T2, T3) are logged. The TF-IDF and MPNet-embedding earnings-text signals
were tested earlier on the same filings (2024 IC +0.118, t 2.89; 2025-26 about 0.003): the
test window has been seen with related methods, which is why the bar is t >= 3.
