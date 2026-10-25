# News Absorption Speed, Intraday News Trading, and LLM-Based Return Prediction

Research status note: this pass was cut short by a tool session limit after about 20 tool calls. URLs marked **[opened]** were fetched and checked during this session. URLs marked **[search-only]** came up in search results with matching title/abstract snippets but could NOT be opened before the limit hit, so treat them as unverified. Several requested items (Chen-Kelly-Xiu numbers, social media and earnings-call evidence, Tetlock 2007 numbers, RavenPack/Benzinga latency studies) are listed under Gaps. They were not invented.

## (a) How quickly is news priced? Large vs small stocks

### Takeaway
Most of the price response to firm news happens within minutes. The LLM evidence (Lopez-Lira & Tang) shows an initial reaction you cannot trade on, which a model reproduces with ~90% hit rates. That is followed by a smaller, tradeable drift lasting about one day, which is concentrated in small caps and negative news. High-frequency decompositions (Jiang, Li & Wang 2021) find the drift continues for several days with no reversal. Stale or repeated news is the exception: it gets overreacted to and then reverses within a week.

### Cited Findings
- Lopez-Lira & Tang separate a **non-tradeable "initial reaction"** from a **tradeable "drift"**. GPT-4 gets ~90% portfolio-day hit rates on the initial reaction. Its scores then predict the later drift, "particularly for smaller-capitalization stocks and negative news." — [arXiv 2304.07619 abstract, v6 Oct 2025](https://arxiv.org/abs/2304.07619) [opened]
- Initial-reaction size for intraday news (same period, non-tradeable): mean +1.33% for positive-scored news and -3.11% for negative-scored news. Initial-reaction hit rate is 93.3% for overnight news and 88.8% for intraday news. — [Lopez-Lira & Tang, full text v6](https://arxiv.org/html/2304.07619v6) [opened]
- Speed: for intraday news, the GPT-4 score coefficient is 0.189 (t=5.12) when predicting the return from 15 minutes after the headline to the close. It falls to 0.116 (t=2.31) for the next day's close-to-close return. The authors read this as "significant value to trade...quickly (within 15 minutes)". In other words, most of the remaining edge decays within the same session. — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6) [opened]
- Small vs large: the small-cap interaction coefficient on overnight drift is 0.404 (t=4.75), so predictability is much stronger in small stocks. The short leg (negative news) earns 26 bps/day (Sharpe 2.01) versus 8 bps/day (Sharpe 0.78) for the long leg. Negative news is absorbed more slowly. — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6) [opened]
- Jiang, Li & Wang ("Pervasive underreaction: Evidence from high-frequency data", JFE 141(2), 2021, pp. 573-599) split daily returns into news-driven and non-news-driven parts using intraday data. After news, prices "drift in the same direction as the initial market response for several days after the news arrival without reversals." A drift strategy "remains profitable after transaction costs." The drift is stronger when investors are distracted and when analysts adjust slowly. — [SSRN 2679614](https://papers.ssrn.com/abstract=2679614) [search-only]; [IDEAS record](https://ideas.repec.org:443/a/eee/jfinec/v141y2021i2p573-599.html) [search-only]
- Tetlock (RFS 2011, "All the News That's Fit to Reprint") measures staleness as a story's textual similarity to the previous 10 stories about the same firm. Returns respond less to stale news, but the return on a stale-news day **negatively predicts the next week's return** (a reversal). The reversal is bigger in stocks with heavy individual-investor trading. — [IDEAS record, RFS 24(5):1481-1512](https://ideas.repec.org/a/oup/rfinst/v24y2011i5p1481-1512.html) [search-only]
- Ke, Kelly & Xiu ("Predicting Returns With Text Data", NBER w26186, 2019) build a supervised sentiment model (SESTM) on Dow Jones Newswires. It has three steps: screen for terms correlated with returns, weight them with a topic model, and score articles by penalized likelihood. The NBER abstract page gives no speed or Sharpe figures. — [NBER w26186](https://www.nber.org/papers/w26186) [opened]
- A Norwegian high-frequency study (20-second VAR, Oslo Stock Exchange, pre-processed news sentiment) found abnormal returns of about -17 bps 90 minutes after negative news. Positive and neutral news showed no significant return effect. Volatility and traded value were already rising (about 0.47-0.48 SD) 5-10 minutes before news arrival, which points to leakage or timestamp lag. — [NHH thesis record](https://openaccess.nhh.no/nhh-xmlui/handle/11250/2678235) [search-only; this is a master's thesis, low weight]

### Inferences
- Rough timeline for US equities: seconds to minutes for the bulk of the move (the initial reaction, which only very fast traders can capture), then the rest of the session, then a smaller drift over 1 to several days. The drift is larger for small caps, negative news, and periods when investors are distracted.
- In the Lopez-Lira & Tang data the remaining edge roughly halves from "15 minutes after the headline" to "next day". That suggests a half-life on the order of hours to one day for an LLM-sentiment signal on fresh news. This is my inference; the paper does not estimate a half-life.
- Stale or rehashed stories should be filtered out or traded as reversals, not momentum.

### Gaps
- Could not open or extract numbers from Tetlock (2007, JF), Gross-Klussmann & Hautsch ("When machines read the news", Reuters NewsScope, second-level reactions; candidate URL https://gfk-cfs.de/media//09_31.pdf), or any study on Dow Jones / Benzinga / RavenPack timestamp latency. No decay half-lives in seconds were verified.
- Jiang-Li-Wang drift sizes, holding periods and cost assumptions were not extracted.

## (b) Do text/news signals survive realistic transaction costs?

### Takeaway
At daily horizons, LLM news signals post high gross Sharpe ratios (about 2-3 for GPT-4 on post-cutoff data). They are very sensitive to costs: profitable at 5-10 bps round-trip and unprofitable at 20 bps. The edge has also decayed fast as LLM use spread (Sharpe went from 6.5 to 1.2 between 2021Q4 and 2024). The multi-day drift (Jiang-Li-Wang) is reported to survive costs.

### Cited Findings
- GPT-4 overnight-news drift strategy, Oct 2021-May 2024 (after GPT-4's Sep 2021 training cutoff): long-short 0.34%/day, annualized Sharpe 2.97, drift hit rate 58%. Intraday-news drift: Sharpe 2.63, hit rate 55%. — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6) [opened]
- Costs: profitable at 5 bps round-trip (cumulative return >300%) and still positive at 10 bps (>100%), but **unprofitable at 20 bps round-trip**. Rebalancing only 25% of the book per day gives Sharpe 1.34 at 10 bps, versus 1.29 with full daily rebalancing. — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6) [opened]
- Decay: strategy Sharpe fell from 6.54 (2021Q4) to 3.68 (2022), 2.33 (2023) and 1.22 (Jan-May 2024). The authors tie this to rising LLM adoption and better price efficiency. — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6); [abstract](https://arxiv.org/abs/2304.07619) [opened]
- Jiang-Li-Wang's high-frequency news-drift strategy "remains profitable after transaction costs" (multi-day horizon). — [SSRN 2679614](https://papers.ssrn.com/abstract=2679614) [search-only]
- Chen, Kelly & Xiu ("Expected Returns and Large Language Models") use contextual LLM embeddings of news in 16 international markets and 13 languages. Search snippets say a strategy trading fresh news alerts gets higher Sharpe ratios. — [Wharton Jacobs Levy slides (Kelly)](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2024/09/Kelly-WhartonJL.pdf) [opened, but the PDF text could not be extracted, so numbers were not verified]; SSRN 4416687 returned HTTP 403.
- DatedGPT-instruct (time-stamped, no look-ahead) reports a 3.20 annualized Sharpe on stock prediction. — [arXiv 2603.11838](https://arxiv.org/abs/2603.11838) [opened; no cost assumptions in the abstract]

### Inferences
- With a 20 bps round-trip break-even, a daily-rebalanced news strategy is only realistic for liquid names with tight spreads, or with partial rebalancing and turnover control. But the alpha sits mostly in small caps and on the short side, where spreads, borrow costs and short constraints are worst. That tension is the main reason to be skeptical about whether it can be captured in practice.
- Multi-day holding (1-5 days) cuts turnover and may survive costs better, but the per-trade edge is smaller. Expect post-2024 Sharpe ratios well below the published figures.

### Gaps
- No verified net-of-cost Sharpe for Chen-Kelly-Xiu, Ke-Kelly-Xiu (SESTM), or Kirtac & Germano (OPT sentiment, Finance Research Letters 2024). The search failed at the session limit.
- No verified cost-inclusive evidence at minute horizons.

## (c) LLM / transformer evidence, look-ahead and memorization problems, remedies

### Takeaway
Bigger LLMs forecast better (GPT-4 > GPT-3.5 > BART/Llama2-70B > BERT/GPT-1/2, which have negative Sharpe). LLMs memorize pre-cutoff financial data, and prompting them to ignore it does not work. Clean evidence therefore needs either post-cutoff test samples or time-stamped models. Studies of time-stamped models disagree on how big the look-ahead premium is: ChronoBERT/ChronoGPT call it "modest", while DatedGPT measures 26.4 bps per SD. Separately, firm identity can distract the model (Glasserman-Lin), so anonymizing headlines helps.

### Cited Findings
- Model scaling (overnight drift Sharpe): GPT-4 2.97; GPT-3.5 1.66; DistilBART-MNLI 1.26; BART-Large 1.05; Llama2-70B 0.97; GPT-1, GPT-2 and BERT negative. Initial-reaction hit rates: GPT-4 and GPT-3.5 93%, DistilBART 87%, BART-Large and Llama2-70B 86%, basic models below 65%. — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6) [opened]
- Supervised embeddings vs zero-shot GPT-4 (drift Sharpe): overnight, GPT-4 2.97 vs OpenAI embeddings 2.53; intraday, 2.63 vs 1.71. MiniLM embeddings get only 1.42 overnight and 0.79 intraday. — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6) [opened]
- Glasserman & Lin (arXiv 2309.17322) separate **look-ahead bias** (the model knows what happened after the news) from a **distraction effect** (general knowledge of the company contaminates the sentiment). Headlines with firm identifiers removed **outperformed** the originals in-sample, so distraction outweighed look-ahead, especially for large, well-known firms. Look-ahead was not a problem out-of-sample. They propose anonymization as a de-biasing step. — [arXiv 2309.17322](https://arxiv.org/abs/2309.17322) [opened]
- Lopez-Lira, Tang & Zhu ("The Memorization Problem", arXiv 2504.14765): LLMs recall exact pre-cutoff economic and financial values. Instructing the model to respect a date and masking both **fail** to stop recall. The problem extends to embeddings. "Post-cutoff, we observe no recall." — [arXiv 2504.14765](https://arxiv.org/abs/2504.14765) [opened]
- Sarkar & Vafa ("Lookahead Bias in Pretrained Language Models", ICML 2025 workshop) test for look-ahead by asking models to predict events that should be unpredictable. They find bias in predicting risk factors from earnings calls and in predicting election winners. Prompting-based fixes have limited effect. They recommend models pretrained only on text from before the analysis period and free of survivorship bias. — [ICML 2025](https://icml.cc/virtual/2025/48685) [opened]
- He, Lv, Manela & Wu ("Chronologically Consistent Large Language Models", arXiv 2502.21206) train ChronoBERT and ChronoGPT only on text available at each date. On news-based return prediction they reach "Sharpe ratios comparable to a much larger Llama model, indicating that lookahead bias is modest." — [arXiv 2502.21206](https://arxiv.org/abs/2502.21206) [opened]
- DatedGPT (arXiv 2603.11838, 2026): twelve 1.3B-parameter models, each trained from scratch on ~100B tokens with strict annual cutoffs from 2013 to 2024. The look-ahead-free instruct model gets Sharpe 3.20. Standard models show a "lookahead premium of 26.4 b.p. per standard deviation," significant at 1%. — [arXiv 2603.11838](https://arxiv.org/abs/2603.11838) [opened]
- Ke-Kelly-Xiu SESTM is the main pre-LLM supervised text baseline, built on Dow Jones Newswires. — [NBER w26186](https://www.nber.org/papers/w26186) [opened]

### Inferences
- The two time-stamped-model studies disagree on how large the look-ahead premium is ("modest" vs 26.4 bps/SD, significant). Treat any in-sample backtest with a frontier LLM as upward-biased by an unknown amount.
- The trustworthy design is to test only on text dated after the model's training cutoff, or to use a model family with point-in-time cutoffs (ChronoGPT, DatedGPT, TimeMachineGPT-style).

### Gaps
- BloombergGPT, FinBERT and TimeMachineGPT (arXiv 2404.18543) results were not verified this session.
- Chen-Kelly-Xiu effect sizes were not verified (SSRN 403, slide PDF unreadable).
- Earnings calls, analyst reports, 8-Ks and social media (StockTwits, Reddit, X): **no sources verified in this session.** The search was blocked by the session limit. The report writer should not draw conclusions about these from this file.

## (d) Concrete methods that worked; intraday feasibility for non-HFT; best practices

### Takeaway
What worked: (1) score each headline per stock with a large LLM (zero-shot YES/NO/UNKNOWN-style prompting in Lopez-Lira & Tang) or with supervised embeddings plus a linear/logistic head; (2) target the **next-period** return after the news is public: open-to-close for overnight news, or 15 minutes after the headline to the close for intraday news; (3) hold long-short for about 1 day, with partial rebalancing to control costs; (4) weight toward small caps and negative news; (5) drop stale news. Trading within the same session, 15 minutes after the news, is still profitable in the published data, so a non-HFT participant can get some of the edge, but not the initial reaction.

### Cited Findings
- Timing design: overnight headlines are traded at the open and held to the close. For intraday headlines, entering 15 minutes after the headline and holding to the close still carries significant predictability (coefficient 0.189, t=5.12). This is the main published evidence that some news edge is reachable at minute latency rather than microsecond latency. — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6) [opened]
- Cost control: partial (25%) daily rebalancing improves net Sharpe at 10 bps (1.34 vs 1.29). — [Lopez-Lira & Tang v6](https://arxiv.org/html/2304.07619v6) [opened]
- Anonymize firm names and tickers before LLM scoring to cut the distraction effect and look-ahead bias. — [Glasserman & Lin](https://arxiv.org/abs/2309.17322) [opened]
- Use post-cutoff test windows; prompts and masks do not stop memorization. — [Lopez-Lira, Tang & Zhu](https://arxiv.org/abs/2504.14765) [opened]
- Use point-in-time pretrained models free of survivorship bias for historical backtests. — [Sarkar & Vafa](https://icml.cc/virtual/2025/48685); [ChronoBERT/ChronoGPT](https://arxiv.org/abs/2502.21206); [DatedGPT](https://arxiv.org/abs/2603.11838) [all opened]
- Filter stale news by textual similarity to the previous 10 stories about the firm, since stale news reverses within a week. — [Tetlock 2011, IDEAS](https://ideas.repec.org/a/oup/rfinst/v24y2011i5p1481-1512.html) [search-only]
- A 2025 study of ChatGPT sentiment on 4,682 dividend-announcement news items (394 S&P 500 firms, 2023-2024, 30-minute intervals) found predictive power mainly in the first two hours after release. This was reported in a search snippet only; the paper's URL and authors were not verified. — no verified URL

### Inferences
- Implications for a daily watchlist pipeline like Atlas: (1) a once-a-day batch pipeline can realistically capture only the **drift** (next open to close, or 1-5 day holds), not the initial reaction; (2) news timestamps must be compared to market hours so overnight news is traded at the next open and nothing is scored on information released after the decision time; (3) evaluate only on dates after the scoring model's training cutoff; (4) include a cost sweep (5/10/20 bps) and a decay-by-year breakdown in any holdout test; (5) expect most alpha in small caps and negative news, so check borrow availability and spreads.
- For intraday reaction (minutes), a retail-latency system (seconds to a minute from headline to order) is plausibly fast enough for the "15 minutes to close" window documented by Lopez-Lira & Tang. That window had decayed by 2024 along with the rest of the strategy.

### Gaps
- No verified evidence on news latency for retail vendor feeds (Benzinga, Alpaca news, etc.), or on how much of the move happens in the first seconds versus the first minutes for large vs small caps.
- Specific IC values were not found in the sources opened. The papers report Sharpe ratios, hit rates and regression coefficients instead.
