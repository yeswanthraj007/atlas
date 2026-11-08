# Result: language-model reader of earnings releases v1 — rejected

Pre-registration: research/preregistered/2026-11-07-llm-reader-v1.md. Trials #84 (T2), #85 (T3),
#86 (T1). RoBERTa-base (training text ends 2019), fine-tuned end to end on 30,809 releases
(to 2022), validated on 5,756 (2023), scored once on 16,647 frozen-test releases
(2024-01 to 2026-09, S&P 1500). Company names and tickers masked. The kept epoch is the
one with the lowest validation loss (epoch 1: 2.3529; epoch 2: 2.3541).

| Target | LLM reader | TF-IDF baseline |
|---|---|---|
| T1 reaction sign (nowcast, not tradable): accuracy / AUC | 50.7% / 0.512 | 54.3% / 0.565 |
| T3 move size vs quarter median: accuracy / AUC | 62.6% / 0.674 | 63.2% / 0.681 |
| T2 20-day drift: IC (t, 33 months) | +0.0033 (0.18) | +0.0041 (0.34) |
| T2 top-minus-bottom decile, gross / net | -1.30% / -1.59% (t -2.15) | +0.29% / +0.03% |

Paired monthly t, LLM minus TF-IDF: T2 IC -0.04; T3 accuracy -1.35.
All four criteria failed. The reader does not beat bag-of-words on any task.

## What it shows
1. The release text alone does not say which way the stock will move (T1 50.7-54.3%): the reaction
   depends on results compared with what the market EXPECTED, and expectations are not in the
   release. The 60% result in the literature (Kim, Muhn & Nikolaev) is for earnings direction
   from financial statements, not for the price reaction.
2. Move size (T3) is about 63% from either model and from the earlier price-only model, so
   it is largely a stock-level property, not a reading skill.
3. A 125M-parameter reader adds nothing over word counts here. Larger models cannot be
   backtested honestly on 2024-2026 (they know the outcomes), so a frontier model can only be
   tested forward in time.
