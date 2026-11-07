# LLMs for return prediction: what the research says (2023-2026) and what it means for Atlas

Written 2026-11-07 from a literature search. Every claim below comes from a source listed at
the end; where the search result gave only a summary, it is marked "(summary)".

## The short version
1. LLMs add value as **readers of text** (news, filings, earnings releases) that turn words
   into numbers. They do **not** add value as forecasters of numeric series.
2. The biggest threat to every published LLM backtest is **look-ahead and memorization**: the
   model has seen the future. Studies that control for it find smaller, but not always zero, effects.
3. Where it works at all, the edge is **short-lived and shrinks as adoption rises**.
4. The most robust positive result is not about returns: **GPT-4 predicts the direction of
   future earnings at about 60%**, against 52.7% for analysts.
5. Live LLM trading experiments show **general intelligence does not transfer to trading skill**.

## Evidence
**Text to returns (works, decays).**
- Lopez-Lira & Tang: ChatGPT scores of news headlines predict out-of-sample next-day returns,
  more among small stocks and after negative news; GPT-1, GPT-2 and BERT could not. A later summary reports
  the strategy's Sharpe ratio falling (to 1.22 by early 2024) as LLM use spread (summary).
- Chen, Kelly & Xiu: embeddings from the last layer of LLMs (BERT, RoBERTa, OPT) predict
  returns better than earlier text methods, in 16 equity markets and 13 languages (summary).
  This is the approach our news model followed.

**Look-ahead and memorization (the central problem).**
- Glasserman & Lin (2023): anonymising headlines removed information but improved in-sample
  results, so the "distraction" effect of company names was larger than look-ahead bias.
- He, Lv, Manela & Wu (2025): ChronoBERT/ChronoGPT, trained only on text available at each date,
  reach a Sharpe ratio comparable to state-of-the-art models in a news task, so look-ahead
  bias was modest there.
- Gao, Jiang & Yan (2026): memorization substantially amplifies apparent predictive power;
  higher memorization raised the marginal effect of the LLM forecast by about 37% of its standalone effect (summary).

**Earnings direction (the strongest accuracy result).**
- Kim, Muhn & Nikolaev (2024): GPT-4, given anonymised, standardised financial statements
  and a chain-of-thought prompt, predicted the direction of future earnings with about 60%
  accuracy against 52.71% for analysts, on par with a purpose-built neural network (about 60%).

**Numbers are not text.**
- Tan et al. (NeurIPS 2024): removing the LLM from three popular LLM-based time-series
  forecasters, or replacing it with a basic attention layer, did not hurt and often helped.
  Our own shootout agrees: Google's TimesFM and Amazon's Chronos were the worst volatility
  forecasters, and nothing predicted returns.

**Live and agent evidence.**
- LiveTradeBench: 21 LLMs over a 50-day live evaluation; high general-benchmark scores did not
  imply better trading outcomes. Alpha Arena gave LLMs real money: styles differed by model
  and results were mixed (summary).
- Trading-R1: a small LLM fine-tuned with reinforcement learning reports strong results on a
  three-month held-out window (June-August 2024) on six large tickers. A short window on six
  stocks cannot establish an edge.

**Complexity.** Kelly, Malamud & Zhou (J. Finance 2024): predictive models with far more parameters
than observations, properly shrunk, forecast returns better than simple ones. It supports using
rich models, with strong regularisation.

## Design principles for an LLM predictor in Atlas
1. **Use the LLM as a reader.** Text in, calibrated probabilities out. No numeric series.
2. **Be chronologically clean.** Pin open-weight models whose training data ends before the test
   window (we use a 2021 model for a 2024-2026 test) and anonymise names and dates. A model from 2025 cannot be
   backtested honestly on 2024: it knows the answers. The only fully clean test is forward in time.
3. **Choose events where text carries information the numbers don't:** earnings releases and
   calls, guidance, 8-K events.
4. **Benchmark against the strongest non-LLM model** (the neural network or tree model that
   already reads the same inputs). The LLM must beat it.
5. **Expect decay.** Test on smaller stocks too, where the published effects are larger, and keep
   a live forward record.
6. **Use the LLM's output as one expert** in the QUORUM design, not as the whole system.

## Sources
- [Can ChatGPT Forecast Stock Price Movements? (Lopez-Lira & Tang)](https://arxiv.org/abs/2304.07619v5)
- [ChatGPT and DeepSeek: Can They Predict the Stock Market and Macroeconomy?](https://arxiv.org/pdf/2502.10008)
- [Expected Returns and Large Language Models (Chen, Kelly & Xiu)](https://www.bohrium.com/paper-details/expected-returns-and-large-language-models/1126027918523236353-2000000)
- [Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT (Glasserman & Lin)](https://arxiv.org/abs/2309.17322)
- [Chronologically Consistent Large Language Models (He, Lv, Manela & Wu)](https://arxiv.org/html/2502.21206v3)
- [Financial Statement Analysis with Large Language Models (Kim, Muhn & Nikolaev)](https://arxiv.org/abs/2407.17866v3)
- [Are Language Models Actually Useful for Time Series Forecasting? (Tan et al.)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract.html)
- [LiveTradeBench: Seeking Real-World Alpha with Large Language Models](https://arxiv.org/html/2511.03628v1)
- [Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning](https://ar5iv.labs.arxiv.org/html/2509.11420)
- [The Virtue of Complexity in Return Prediction (Kelly, Malamud & Zhou)](https://www.nber.org/system/files/working_papers/w30217/revisions/w30217.rev0.pdf)
- [Predictability of Post-Earnings Announcement Drift with Textual and Contextual Factors of Earnings Calls](https://www.aminer.org/pub/656da6f5939a5f4082df1b33/predictability-of-post-earnings-announcement-drift-with-textual-and-contextual-factors-of)
