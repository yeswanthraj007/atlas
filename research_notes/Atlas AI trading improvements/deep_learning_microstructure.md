# Deep Learning for Short-Horizon / Intraday Prediction, Foundation Models, and ML Volatility Forecasting

Scope note: research stopped early because the web-search session limit was hit partway through. Every URL below was opened (via WebFetch or curl/PDF download) during this session. Some planned sources (Nagel's critique of "virtue of complexity", Barroso & Detzel on vol-managed costs, TransLOB/DeepLOB-Attention, deep-RL portfolio papers, Kronos) could not be verified and are listed under Gaps rather than cited.

## 1. Limit order book (LOB) deep learning: what accuracy, at what horizon, and does it generalize or survive costs?

### Takeaway
On the FI-2010 benchmark, DeepLOB-style models report about 75-85% accuracy/F1 at horizons of 10-100 events. When the 2023 LOBCAST benchmark retrained 15 such models on newer NASDAQ data, F1 fell to roughly 48-61%. Separately, a 2024 study found that high forecast accuracy "does not necessarily correspond to actionable trading signals." None of the papers checked here shows profits net of costs for a non-co-located trader.

### Cited Findings
- DeepLOB (Zhang, Zohren & Roberts) combines CNN filters, which read the structure of the order book, with LSTM modules, which capture longer time dependencies. It was tested on FI-2010 and on one year of London Stock Exchange (LSE) quotes, and the authors claim it "translates well to instruments which were not part of the training set" — [arXiv 1808.03668](https://arxiv.org/abs/1808.03668)
- DeepLOB results on FI-2010 (setup 2) — [arXiv PDF 1808.03668](https://arxiv.org/pdf/1808.03668):
  - k=10 events: accuracy 84.47%, F1 83.40%, against C(TABL) at 84.70% accuracy and 77.63% F1.
  - k=20: accuracy 74.85%, F1 72.82%.
  - In setup 1: k=50 F1 74.96%; k=100 accuracy 76.66%, F1 76.58%.
- FI-2010 itself has known weaknesses. It covers 5 Nasdaq Nordic stocks over 10 consecutive days in 2010, from a less liquid market. The DeepLOB authors themselves say it is "arguably not sufficient to fully verify the robustness of an algorithm", and that its smoothed labelling produces inconsistent signals and "larger transaction costs". LSE out-of-sample testing used 5 stocks over a 3-month test period — [arXiv PDF 1808.03668](https://arxiv.org/pdf/1808.03668)
- DeepLOB's own trading simulation "use[s] mid-prices without transaction costs". The authors concede this assumption is not reasonable for a standalone strategy and use the simulation only for relative model comparison — [arXiv PDF 1808.03668](https://arxiv.org/pdf/1808.03668)
- The LOBCAST benchmark (2023, arXiv 2308.01915) covers 15 deep-learning LOB models in an open-source framework with profit analysis. Its headline: "all models exhibit a significant performance drop when exposed to new data" — [arXiv 2308.01915](https://arxiv.org/abs/2308.01915)
- LOBCAST detailed numbers — [arXiv HTML 2308.01915](https://arxiv.org/html/2308.01915):
  - Best model on FI-2010, BINCTABL: F1 81.0% (k=1), 71.2% (k=2), 88.1% (k=5), 92.1% (k=10).
  - On the new NASDAQ LOB-2021/2022 data, models scored F1 of only about 48-61%.
  - BINCTABL's F1 fell by about 19.6% on average across horizons.
  - Observations are sampled every 10 events, so the horizons are short event-time windows.
  - On backtesting, it notes "models' profitability is far from guaranteed".
- "Deep Limit Order Book Forecasting" (2024) studies a mixed set of NASDAQ stocks — [arXiv 2403.09267](https://arxiv.org/abs/2403.09267):
  - Each stock's microstructure (tick size, spread) affects how well deep learning works.
  - High forecast accuracy "does not necessarily correspond to actionable trading signals".
  - It proposes judging forecasts by the probability of correctly predicting complete transactions instead.

### Inferences
- Reported FI-2010 accuracies (80-90%+) are mostly an artefact of the dataset and its labels. The roughly 50-60% F1 seen on recent, liquid NASDAQ data is the realistic level, and it applies to horizons of tens to hundreds of events, i.e. seconds or less in liquid names.
- Crossing the spread usually costs more than a mid-price move that small, so the edge disappears for anyone who has to take liquidity. This is especially true for a retail or non-co-located trader with tens to hundreds of milliseconds of latency and only delayed or consolidated data. Signals like these mainly help execution timing (passive vs. aggressive order placement), not standalone alpha.
- Large-tick stocks give more predictable mid-price changes but larger relative spreads. The predictability and the cost come from the same source.

### Gaps
- No verified source for TransLOB / DeepLOB-Attention accuracy figures, or for LOBench (2024-25).
- No peer-reviewed study found that shows deep LOB signals are profitable net of fees and latency for a non-co-located participant.
- I did not extract exact model-by-model LOB-2021 F1 values or the LOBCAST backtest P&L numbers (only summarized).

## 2. Transformer and foundation models (TSFMs) for financial time series: returns vs. volatility

### Takeaway
Off-the-shelf foundation models (Chronos, TimesFM, Moirai, TimeGPT) are not reliably better than a random walk or simple ML for return forecasting. Pre-training on financial data helps, but the reported Sharpe ratios ignore transaction costs. For realized volatility the evidence is mixed: no foundation model dominates HAR, and a combination of the small TTM model with log-HAR is the most robust.

### Cited Findings
- Rahimikia, Ni & Wang (2025), "Re(Visiting) Time Series Foundation Models in Finance" — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1):
  - Setup: about 2 billion daily observations from 94 countries; out-of-sample period 2001-2023; 18.1M U.S. daily excess returns on about 10,000 securities.
  - Zero-shot results:
    - Chronos-large: R² of -1.37%, directional accuracy 51%.
    - TimesFM-500M: R² of -2.80%.
  - Fine-tuning mostly made things worse.
  - Pre-training from scratch on financial data helped a lot. For example, Chronos-small reached about 36.8% annualized return and a 5.42 Sharpe on decile long-short portfolios (window 512).
  - CatBoost was still the best model: 46.5% return, 6.79 Sharpe.
  - Transaction costs were not included.
- Noguer i Alonso & Pereira Franklin (2026) compare TimeGPT, TimesFM-2.5, Moirai-2.0, Chronos and Chronos-2 with from-scratch NBEATS, NHITS, PatchTST, iTransformer and KAN, using returns on AAPL, AMZN, GOOG, JPM and META — [arXiv 2606.27100](https://arxiv.org/abs/2606.27100):
  - Pretrained models win 8 of 10 tasks.
  - But "gains over the random-walk benchmark are small and sparse": only 2 model-asset pairs are significant in Diebold-Mariano tests.
  - The authors conclude foundation models are "not universal engines for statistically reliable alpha generation".
- Brini (2026) tests 9 zero-shot foundation models against 8 econometric models (HAR family) for realized volatility on 50 assets (equities, FX, futures), using the VOLARE data — [arXiv 2607.05291](https://arxiv.org/abs/2607.05291):
  - There is no uniform gain. Pooled losses favour the foundation models only because a few outlier assets dominate.
  - Only Tiny Time Mixers (TTM) beats the benchmark at every horizon, and only "by a narrow margin".
  - An equal-weight TTM + log-HAR combination is in the Model Confidence Set for 98-100% of assets.
  - At short horizons the foundation models' advantage reflects better scaling, not better dynamics. Only monthly forecasts show genuine information gains.

### Inferences
- Univariate foundation models have no special edge on near-random-walk return series. Their realistic role is volatility and scale forecasting, ideally combined with HAR, not directional alpha.
- Sharpe ratios of 5-7 on daily decile long-short portfolios with no costs, across about 10,000 securities including micro-caps, are not credible as implementable figures. Section 4 (JKMP) explains why.

### Gaps
- Kronos, a financial-specific K-line foundation model (arXiv 2508.02739), surfaced in search but I did not open it. Lag-Llama and Moirai original papers were not verified.
- No verified evidence on foundation models for intraday or LOB horizons.

## 3. ML for volatility forecasting (HAR vs. ML) and volatility-managed portfolios

### Takeaway
ML and neural networks can beat HAR modestly in realized-variance forecasting. A careful 1,455-stock study shows, however, that a properly windowed HAR beats ML when both use the same inputs. Volatility-managed portfolios show in-sample alpha (Moreira & Muir), but real-time versions generally underperform the unmanaged portfolios (Cederburg et al.), before any trading-cost argument.

### Cited Findings
- Christensen, Siggaard & Veliyev, CREATES RP 2021-03, later published in the Journal of Financial Econometrics (2023) — [CREATES PDF](https://repec.econ.au.dk/repec/creates/rp/21/rp21_03.pdf):
  - Task: one-day-ahead realized variance for 29 DJIA stocks over 2001-2017, built from 5-minute returns.
  - Split: 70/10/20 train/validation/test, giving 847 test days.
  - Off-the-shelf ML (regularization, trees, neural networks) beats HAR even when the only inputs are the daily, weekly and monthly RV lags.
  - ML gets more out of extra predictors (firm characteristics, macro indicators) than an extended HAR (HAR-X) does.
- Search snippets (not verified in the full text) also say that the gains grow at longer horizons and that neural networks cut MSE by up to 11.8% — via the same [CREATES PDF](https://repec.econ.au.dk/repec/creates/rp/21/rp21_03.pdf). Treat the 11.8% as unverified.
- Audrino & Chassot, "HARd to Beat" — [arXiv 2406.08041](https://arxiv.org/abs/2406.08041):
  - On 1,455 stocks with the same predictors (RV, VIX), HAR beats ML when it is properly fitted.
  - What decides the result is the fitting scheme (training window size, re-estimation frequency), not model class.
  - HAR is also far cheaper to run and easier to interpret.
- Moreira & Muir (JF 2017; NBER w22208) — [NBER w22208](https://www.nber.org/papers/w22208):
  - Scaling exposure down when volatility is high produces "large alphas" and higher Sharpe ratios.
  - This holds for the market, value, momentum, profitability, ROE and investment factors, and for the currency carry trade.
- Cederburg, O'Doherty, Wang & Yan (JFE 138, 2020) — [JFE paper PDF](https://www.lehigh.edu/~xuy219/research/COWY.pdf):
  - Across 103 equity strategies, volatility-managed portfolios "do not systematically outperform" the unmanaged versions in direct comparisons.
  - Spanning-regression alphas are positive (the market's is 4.63% a year), but the strategies they imply are not implementable in real time.
  - Out-of-sample versions earn lower certainty-equivalent returns and Sharpe ratios than the unmanaged portfolios. The cause is structural instability in the regression parameters.

### Inferences
- For a risk overlay (position sizing, stops, volatility targeting), a well-specified HAR/log-HAR on 5-minute RV is the right baseline. ML or foundation models are worth adding only if they beat it in a rolling-window evaluation, and the expected gain is small (single-digit % MSE).
- Volatility scaling is better treated as risk control (drawdown and tail reduction) than as an alpha source. Its Sharpe benefit is fragile out of sample even before turnover costs.

### Gaps
- Not verified: Bucci (2020) on neural-network volatility forecasting, and Barroso & Detzel (transaction costs and limits to arbitrage in vol-managed portfolios). The latter is directly relevant to cost survival but unconfirmed here.
- No verified numbers on turnover-cost breakevens for volatility management.

## 4. Cross-sectional ML return prediction (daily/monthly) and transaction costs

### Takeaway
Trees and neural networks beat linear models for cross-sectional prediction, driven by momentum, liquidity and volatility signals (GKX 2020). Jensen, Kelly, Malamud & Pedersen show that cost-agnostic ML portfolios lean on fleeting, small-scale characteristics and earn poor net returns, and that cost-aware learning of portfolio weights is needed. "Virtue of complexity" (KMZ, JF 2024) argues for over-parameterized models; I could not verify the critiques of it in this session.

### Cited Findings
- Gu, Kelly & Xiu, NBER w25398 (RFS 2020) — [NBER w25398](https://www.nber.org/papers/w25398):
  - Trees and neural networks perform best because they capture nonlinear interactions between predictors.
  - All methods agree on the dominant signals: variations on momentum, liquidity and volatility.
  - In some cases the ML strategies double the performance of regression-based ones.
- Jensen, Kelly, Malamud & Pedersen, "Machine Learning and the Implementable Efficient Frontier" (RFS) — [SFI page](https://www.sfi.ch/en/publications/machine-learning-and-the-implementable-efficient-frontier):
  - Strategies should be judged by net-of-trading-cost return at each level of risk.
  - Cost-agnostic ML leads to "excessive reliance on fleeting small-scale characteristics, resulting in poor net returns".
  - Learning portfolio weights directly, with an economic objective and cost-aware optimization, gives a better frontier and a measure of "economic feature importance".
- Kelly, Malamud & Zhou, "The Virtue of Complexity in Return Prediction" (JF 79(1), 2024) — [IDEAS/RePEc](https://ideas.repec.org/a/bla/jfinan/v79y2024i1p459-503.html):
  - They prove theoretically that simple models "severely understate return predictability" compared with models whose parameter count exceeds the number of observations.
  - They document this empirically for U.S. market-return timing.

### Inferences
- For a daily or weekly watchlist system, the practical lesson from JKMP is to penalize turnover and small-cap dependence directly in the objective. Ranking by raw predicted return then trading the top decile is exactly the cost-agnostic setup they show fails net of costs.
- Momentum, liquidity and volatility features dominate, so cheap features from daily bars capture most of what is learnable. Exotic deep architectures add little at these horizons.

### Gaps
- Nagel's critique ("Seemingly Virtuous Complexity in Return Prediction") and other replication critiques of KMZ: the search hit the session limit before I could verify them.
- Out-of-sample decay after publication (McLean-Pontiff-style) for ML signals: not verified here.
- Exact JKMP net Sharpe numbers: not extracted.

## 5. Reinforcement learning for execution and trading

### Takeaway
The strongest RL evidence is in execution, not alpha. Nevmyvaka, Feng & Kearns (ICML 2006) cut execution cost relative to submit-and-leave limit-order baselines by about 27-36% (up to about 50% in relative terms) using NASDAQ order-book data. I could not verify the more recent deep-RL portfolio results, which are generally evaluated in simulation without market impact.

### Cited Findings
- Nevmyvaka, Feng & Kearns, "Reinforcement Learning for Optimized Trade Execution" (ICML 2006) — [UPenn PDF](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf):
  - Data: 1.5 years of millisecond NASDAQ limit-order data.
  - Method: Q-learning combined with dynamic programming, with state built from time remaining, inventory remaining and market variables.
  - Results: up to 50% relative improvement. Improvement over optimized submit-and-leave ranged from 27.16% to 35.50% (AMZN, NVDA, QCOM; 2- and 8-minute horizons; 5K/10K shares).
- Caveat on evaluation: the paper replays historical order books, so the agent's own orders do not change later book dynamics. This is the standard limitation of replay backtests — inferred from the setup described in the [UPenn PDF](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf).

### Inferences
- For a non-co-located retail system, RL's realistic use is order placement (limit vs. marketable, slicing) on fills that are large relative to displayed depth. For small retail sizes the possible savings are a fraction of the spread, and a simple limit-at-mid or near-touch rule probably captures most of it.
- Deep-RL portfolio papers that report high Sharpe ratios in simulation should be treated as unproven unless they include costs, market impact and walk-forward testing.

### Gaps
- Not verified: deep-RL portfolio or trading papers (e.g., Jiang et al. 2017, FinRL), recent deep-RL execution surveys (2022-2025), and any out-of-sample live results. The search session limit stopped this part.
