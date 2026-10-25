# How Top Investment Institutions Use ML, NLP, LLMs and Alternative Data (public disclosures)

Scope note: Every URL cited below was opened during this research (Oct 2026). Items seen only in search snippets, or known from background knowledge but not verified, are put under Gaps or marked "[unverified]". "Official" = the firm's own website or named executives speaking on the record; "press" = third-party reporting.

## 1. What data types and model families does each firm say it uses?

### Takeaway
The disclosures that are officially stated and specific centre on **text/NLP (especially earnings-call transcripts)** and, more recently, **LLM agents that automate the research loop** (Man Group AlphaGPT, Bridgewater AIA, AQR). Firms say much less in public about satellite or card data, model architectures and holding periods. Generalist LLMs are mostly described as productivity tools, not alpha sources (Citadel, Goldman Sachs).

### Cited Findings
**BlackRock (Systematic) - official**
- Article dated 29 Jul 2024 by Raffaele Savi, Jeff Shen, Yaki Tsaig and Taylor Dufour. BlackRock Systematic trained a proprietary model on "more than 400,000 earnings call transcripts covering over 17,000 public firms" together with two decades of market data. The goal is to "learn an association between earnings call text and subsequent market reaction". They benchmarked it against OpenAI GPT models on 500 Q1-2024 earnings calls — [BlackRock: AI in investing](https://www.blackrock.com/institutions/en-us/insights/ai-investing)
- The same article lists text sources: analyst reports, earnings-call transcripts, news and social media. It argues for LLMs fine-tuned on narrow, curated datasets for specific tasks (e.g., forecasting the post-earnings reaction) over general-purpose models. It describes a "Thematic Robot" that builds equity baskets around themes quickly, and says human PM oversight remains "crucial" — [BlackRock](https://www.blackrock.com/institutions/en-us/insights/ai-investing)
- BlackRock's retail-facing version is generic. It frames systematic edge as "informational advantage" rather than risk premia and gives no data types, signal counts or holding periods — [BlackRock individual insights](https://www.blackrock.com/us/individual/insights/ai-investing)

**Man Group / Man Numeric - official + press**
- Man Group article (Oct 2025) by Ziang Fang and Jason Moore. AlphaGPT is "a proprietary agentic AI research workflow" with three LLM components: (1) idea or hypothesis generation, producing "dozens of viable concepts within minutes"; (2) implementation, which turns ideas into production-grade Python; (3) evaluation against statistical and economic criteria. It works best so far in systematic equities, scales poorly across asset classes, and cannot run "unsupervised just yet" — [Man Group: What AI can do for alpha](https://www.man.com/insights/what-ai-can-do-for-alpha)
- Press (Hedgeweek, citing Bloomberg, 11 Jul 2025): AlphaGPT had produced "several dozen" signals approved for live trading. Hallucinations and inconsistent outputs persist. Ziang Fang: "I wouldn't call it autopilot because we have a lot of checkpoints in place" — [Hedgeweek](https://hedgeweek.com/news/man-group-deploys-agentic-ai-for-quant-signal-discovery)

**Bridgewater (AIA Labs / ML fund) - press, quoting executives**
- Launched July 2024 with about $2bn from more than six clients. ML is "the primary basis" for decisions. It uses proprietary technology built over more than a decade plus models from OpenAI, Anthropic and Perplexity. It was tested from late 2023 on a roughly $100m sleeve of Pure Alpha. Humans still handle risk, data acquisition and execution — [Fortune, 1 Jul 2024](https://fortune.com/2024/07/01/bridgewater-2-billion-fund-machine-learning-decision-making-openai-anthropic-perplexity)
- Greg Jensen on LLM limits: "Large language models have the problem of hallucination. They don't know what greed is, what fear is, what the likely cause-and-effect relationships are." — [Fortune](https://fortune.com/2024/07/01/bridgewater-2-billion-fund-machine-learning-decision-making-openai-anthropic-perplexity)

**AQR - executives on the record + affiliated academic research**
- Cliff Asness (May 2024) called AI "still just statistics" and "evolutionary, not revolutionary". He named three uses: (1) generating trading signals, with NLP the most-used application; (2) weighting and combining signals systematically, which "doesn't seem to have the same degree of trending"; (3) productivity such as code writing. He also criticised "AI washing" — [Hedgeweek](https://www.hedgeweek.com/ai-is-still-just-statistics-says-aqrs-cliff-asness/)

**Two Sigma - official (thin)**
- Two Sigma acquired Hivemind's software (24 Aug 2022). Hivemind combines automation with distributed human labelling to turn semi- and unstructured sources into machine-readable datasets. CTO Jeff Wecker called alternative data workflows "a key input in our modeling and machine learning practices" — [Two Sigma](https://www.twosigma.com/articles/two-sigma-announces-acquisition-of-hivemind-software/)

**Citadel - executive on the record**
- Ken Griffin (16 Oct 2025, JPMorgan Robin Hood Investors Conference) said GenAI helps productivity, "but for uncovering alpha it just falls short". He added that it has not replaced Citadel's core research and is unlikely to change the industry profoundly — [Hedgeweek](https://www.hedgeweek.com/generative-ai-fails-to-deliver-alpha-for-hedge-funds-says-griffin/)

**Goldman Sachs - press**
- The GS AI Assistant was rolled out firmwide (about 46,000 staff) in June 2025 after a pilot with more than 10,000 employees. It includes a developer copilot, translation and a "Banker Copilot", and gives access to ChatGPT and Gemini behind Goldman's firewall. Uses are document summarisation, drafting and data analysis. Nothing in this coverage is about alpha generation — [Storyboard18](https://storyboard18.com/digital/goldman-sachs-rolls-out-generative-ai-assistant-firmwide-71562.htm)

**Renaissance Technologies - third-party only (the firm publishes nothing)**
- According to Zuckerman, as summarised by Quartr, success came from carefully curated, comprehensive data, early use of ML/statistical models, aggressive sizing when odds were favourable, PhD hiring, and leverage averaging about 12.5x (sometimes 20x) — [Quartr](https://quartr.com/insights/company-research/renaissance-technologies-and-the-medallion-fund)

### Inferences
- Across the firms that disclose concrete methods (BlackRock, AQR, Man Group), the common pattern is **domain-specific NLP on corporate text, labelled by the subsequent price reaction**. That is the most replicable idea for a solo builder: label transcripts or news with forward returns and train a small specialised model rather than prompting a general LLM.
- The 2024-2025 shift is from "LLM as signal" to "**LLM as research assistant**" (generate hypothesis, code it, backtest it). The same firms wrap this in human committees and strict statistical gates.

### Gaps
- None of the firms publicly details satellite, card or geolocation datasets, specific model families (GBM vs. NN), or feature counts. Two Sigma's articles site was not mined beyond Hivemind.
- Not covered or not verified: Aladdin / Aladdin Copilot technical detail, BlackRock AI Labs, Man AHL's specific ML papers (e.g., Anthony Ledford's work on ML in trading), Citadel Securities' market-making technology, D.E. Shaw, Point72/Cubist, WorldQuant (incl. its "alpha factory"/BRAIN crowdsourcing), Millennium. These remain open because of the tool-call budget.
- Search snippets (unopened, so unverified) claim Asness said in June 2025 that earnings-calls-to-stock-recommendations is "pure machine learning" and has "worked out quite nicely". Bloomberg (Dec 2024) reportedly said AQR uses AI to combine and weight factors and to generate signals from text.

## 2. What do their papers/blogs say works and does not work for return prediction?

### Takeaway
The AQR-affiliated academic line pulls two ways. "Can Machines Learn Finance?" stresses that finance is a hostile domain for ML (small data, low signal-to-noise, nonstationarity), so economic theory and human expertise are needed. "The Virtue of Complexity" argues that heavily over-parameterised but properly shrunk models outperform simple ones. Practitioners report the most success in NLP signals and signal combination, and the least in using generalist LLMs as direct alpha sources.

### Cited Findings
- Israel, Kelly and Moskowitz, "Can Machines 'Learn' Finance?" (30 Apr 2020, Journal of Investment Management; 2020 Harry M. Markowitz Award). ML for asset management "faces a unique set of challenges that differ markedly from other domains where machine learning has excelled". It stresses "the importance of economic theory and human expertise" — [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Can-Machines-Learn-Finance)
- Kelly, Malamud and Zhou, "The Virtue of Complexity in Return Prediction", Journal of Finance 79(1):459-503, Jan 2024. "Simple models severely understate return predictability compared to 'complex' models in which the number of parameters exceeds the number of observations". Supported by theory and by US equity market-timing evidence — [SFI](https://www.sfi.ch/en/publications/the-virtue-of-complexity-in-return-prediction); full text at [Yale-hosted PDF](https://economics.yale.edu/sites/default/files/2024-01/The%20Journal%20of%20Finance%20-%202023%20-%20KELLY%20-%20The%20Virtue%20of%20Complexity%20in%20Return%20Prediction%20%281%29.pdf). The PDF could not be text-parsed, so its specific Sharpe/R² numbers are not extracted here.
- AQR (Asness, May 2024): NLP signals and ML-based signal weighting are where ML has helped. The improvement is "evolutionary, not revolutionary" — [Hedgeweek](https://www.hedgeweek.com/ai-is-still-just-statistics-says-aqrs-cliff-asness/)
- Citadel (Griffin, Oct 2025): GenAI "falls short" for alpha — [Hedgeweek](https://www.hedgeweek.com/generative-ai-fails-to-deliver-alpha-for-hedge-funds-says-griffin/)
- Bridgewater (Jensen): LLMs lack causal understanding and hallucinate, so they are combined with proprietary causal and statistical models — [Fortune](https://fortune.com/2024/07/01/bridgewater-2-billion-fund-machine-learning-decision-making-openai-anthropic-perplexity)
- Man Group: agentic LLM research works for equity signal discovery but not yet across asset classes or without supervision — [Man Group](https://www.man.com/insights/what-ai-can-do-for-alpha)
- BlackRock: narrow, fine-tuned models trained on curated data are preferred over generic LLMs for forecasting tasks — [BlackRock](https://www.blackrock.com/institutions/en-us/insights/ai-investing)

### Inferences
- "Virtue of complexity" does not mean "anything goes". The result depends on heavy ridge-style shrinkage and on many random nonlinear features. For a solo builder, this argues for regularised wide models (e.g., ridge on many random features, or well-regularised GBMs) over hand-tuned narrow ones, always judged out of sample.
- The convergence of Citadel, Bridgewater and AQR suggests that off-the-shelf LLM "stock picking" is not a durable edge. The value lies in text-to-feature extraction and research automation.

### Gaps
- No verified Man AHL paper on ML pitfalls was opened. The search surfaced only a 2016 Bloomberg piece on Man AHL's Anthony Ledford being sceptical of ML hype (pre-2018, unopened).
- Not found or verified: firm-published material on alt-data decay or crowding (e.g., AQR or Two Sigma pieces on factor crowding).

## 3. Holding periods, capacity, and speed/market-making vs prediction

### Takeaway
Almost no firm publishes holding periods or capacity figures. The only quantitative capacity evidence is third-party: Renaissance's Medallion is capped at about $15bn and distributes profits, which implies a strategy limited by capacity and edge. GenAI is described as mid-frequency research tooling, not as a speed edge.

### Cited Findings
- Medallion is "capped at ~$15 billion with annual distributions". Its stated net return for 1988-2022 is 39.9% CAGR (90,129x), which Quartr calls "highly theoretical" — [Quartr](https://quartr.com/insights/company-research/renaissance-technologies-and-the-medallion-fund)
- Bloomberg data, reported in 2017 (older than 2018): Medallion made about $55bn over 29 years, with "average returns after fees of an astounding 40 percent" — [WealthManagement.com](https://www.wealthmanagement.com/alternative-investments/renaissance-mints-another-billionaire-with-two-more-on-the-cusp)
- Bridgewater's ML fund is macro: its test scenarios were tariffs after a Trump win and Fed QT effects on bonds. That implies macro horizons, not HFT — [Fortune](https://fortune.com/2024/07/01/bridgewater-2-billion-fund-machine-learning-decision-making-openai-anthropic-perplexity)
- BlackRock's flagship NLP model targets the post-earnings-call market reaction, i.e., an event-driven horizon of days to weeks; the exact horizon is not stated — [BlackRock](https://www.blackrock.com/institutions/en-us/insights/ai-investing)

### Inferences
- A solo builder cannot compete on speed or market-making (Citadel Securities, Renaissance's short-horizon book). Disclosed firm activity suggests the accessible edge is event-driven text signals and multi-day to multi-week horizons, where capacity limits matter less at small size.

### Gaps
- Not verified: holding periods for AQR, Two Sigma, Man AHL or BlackRock Systematic; Citadel Securities' share of US equity volume; the split of Renaissance's P&L between short-term statistical arbitrage and longer-horizon positions.

## 4. How they validate models and guard against overfitting and look-ahead (incl. LLM look-ahead)

### Takeaway
The firms describe process controls (pre-specified methodology, investment committees, code review, logging, economic-rationale checks) more than specific statistics. Academic work from 2026 shows that LLM look-ahead bias is real and measurable. It can be removed with time-sliced pretrained models.

### Cited Findings
- Man Group applies the "same stringent research methodology that has historically prevented human-driven p-hacking" to AI-generated ideas. Its controls are:
  - dual-track validation: the Investment Committee reviews hypotheses and technology teams review the code;
  - consistency checks between the generation, implementation and evaluation stages;
  - every decision logged for human review.

  [Man Group](https://www.man.com/insights/what-ai-can-do-for-alpha)
- Each AI-produced signal must pass statistical-significance tests, risk assessment and an economic-rationale check before live deployment — [Hedgeweek/Bloomberg](https://hedgeweek.com/news/man-group-deploys-agentic-ai-for-quant-signal-discovery)
- Bridgewater ran a live paper-scale sleeve (about $100m of Pure Alpha) from late 2023 before launching the $2bn fund, i.e., live out-of-sample incubation — [Fortune](https://fortune.com/2024/07/01/bridgewater-2-billion-fund-machine-learning-decision-making-openai-anthropic-perplexity)
- BlackRock benchmarked its specialised model against GPT on 500 Q1-2024 earnings calls. Q1 2024 may be after training cut-offs depending on the model, which makes it a semi-out-of-sample test — [BlackRock](https://www.blackrock.com/institutions/en-us/insights/ai-investing)
- AQR / Israel, Kelly and Moskowitz: economic theory and human expertise are needed to constrain ML in a low signal-to-noise domain — [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Can-Machines-Learn-Finance)
- LLM look-ahead, academic. DatedGPT (Yan, Tang, Gao, Jiang, Lu; arXiv Mar 2026, revised Jul 2026) trained twelve 1.3B-parameter models with strict annual cut-offs from 2013 to 2024:
  - perplexity probing confirms each model's knowledge stops at its cut-off;
  - on 61,000 firm-day news headlines, the bias-free model still earned an annualised Sharpe of 3.20;
  - models that had seen outcome data showed a "lookahead premium" of 26.4 bp per standard deviation, significant at 1%.

  [arXiv 2603.11838](https://arxiv.org/abs/2603.11838)

### Inferences
- Practical protocol for a solo builder, consistent with the above:
  1. Pre-register the hypothesis and an economic rationale before testing.
  2. Count and penalise the number of trials (the Man Group p-hacking framing).
  3. Freeze a holdout and paper-trade before committing capital (the Bridgewater sleeve).
  4. For any LLM-derived feature, only trust backtests on dates after the model's training cut-off, or use time-sliced models such as DatedGPT. Otherwise the measured edge partly comes from memorised outcomes.

### Gaps
- Not opened (403): Arnott, Harvey and Markowitz, "A Backtesting Protocol in the Era of Machine Learning" (SSRN 3275654, about 2018-2019). The ResearchGate/SSRN abstract was unreachable, so it is not cited for content.
- No firm-published, quantified overfitting-haircut methodology was found (e.g., deflated Sharpe usage at a named firm).
- A related arXiv paper surfaced but was not opened: "Summoning the Oracle to Slay It" (FinCAD, May 2026, arXiv 2605.24564), which proposes inference-time suppression of LLM memory.

## 5. Public performance figures and hit rates

### Takeaway
The commonly cited Renaissance "50.75%" hit-rate line is a Robert Mercer quote relayed via Zuckerman's 2019 book. It is not an official disclosure. AI-specific performance figures are essentially undisclosed; the closest is "several dozen" live signals at Man Numeric.

### Cited Findings
- Mercer: "We're right 50.75% of the time... but we're 100% right 50.75% of the time, you can make billions that way." Attributed via Gregory Zuckerman's *The Man Who Solved the Market* (2019) — [Quartr](https://quartr.com/insights/company-research/renaissance-technologies-and-the-medallion-fund). This is third-party relay, not a firm statement.
- Medallion's figure of about 39.9% net CAGR for 1988-2022 is third-party, built from Zuckerman and Bloomberg data — [Quartr](https://quartr.com/insights/company-research/renaissance-technologies-and-the-medallion-fund); Bloomberg's 2017 figure is about 40% average after fees — [WealthManagement.com](https://www.wealthmanagement.com/alternative-investments/renaissance-mints-another-billionaire-with-two-more-on-the-cusp)
- Man Numeric: "several dozen" AlphaGPT signals approved for live trading as of Jul 2025. No P&L disclosed — [Hedgeweek](https://hedgeweek.com/news/man-group-deploys-agentic-ai-for-quant-signal-discovery)
- AQR: Asness says ML signal weighting shows less trending (more stable) behaviour. No numbers given — [Hedgeweek](https://www.hedgeweek.com/ai-is-still-just-statistics-says-aqrs-cliff-asness/)
- Academic benchmark: a look-ahead-free LLM news signal earned Sharpe 3.20 gross in backtest. This is not a firm result and is likely before costs — [arXiv](https://arxiv.org/abs/2603.11838)

### Inferences
- A 50.75% hit rate means the edge comes from **breadth (very many independent bets), sizing and cost control**, not accuracy. A solo builder should judge signals on information coefficient × breadth after costs, not on directional accuracy.

### Gaps
- Bridgewater ML fund returns since its July 2024 launch were not found or verified. No verified post-2018 performance figures for AQR or Two Sigma ML strategies.
- Not verified: Renaissance's public fund (RIEF/RIDA) performance versus Medallion.
