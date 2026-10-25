# Equity Return Signals and Alternative Data: Replication, Decay, Costs, Implementability

Verification legend: [V] = URL opened and content read in this session. [S] = number taken from a search-result snippet; the primary page was blocked (HTTP 403) or returned unreadable PDF, so the number is not independently confirmed. Tool budget (~15 calls) was hit; gaps are listed per section.

## (a) What does the anomaly/factor replication literature find (HXZ, Chen-Zimmermann, JKP, Harvey-Liu-Zhu)?

### Takeaway
The literature splits on framing but agrees on substance: with microcaps neutralized (NYSE breakpoints, value weights) and a t>3 hurdle, most published anomalies are weak or insignificant (HXZ: 64-85% fail); yet the core themes (value, momentum, quality/profitability, low risk, investment, earnings-related) do replicate and hold up out of sample internationally (JKP). Replication is not the binding constraint; costs and decay are.

### Cited Findings
- Hou, Xue & Zhang ("Replicating Anomalies", NBER w23394; RFS 2020): of 447 anomaly variables, 286 (64%) are insignificant at 5%, including 95 of 102 liquidity variables (93%); with a t>3 cutoff, 380 (85%) are insignificant. Of the 161 significant ones, the q-factor model leaves 115 alphas insignificant (150 with t<3). Even surviving anomalies are usually much smaller than originally reported. The key step was NYSE breakpoints plus value-weighted returns to cut microcap influence. Their conclusion: markets are "more efficient than previously recognized." — [NBER w23394](https://www.nber.org/papers/w23394) [V]
- Jensen, Kelly & Pedersen ("Is There a Replication Crisis in Finance?", JF 2023, 78(5):2465-2518): a Bayesian model of factor replication finds most factors (1) replicate, (2) fall into 13 themes, most of which are significant parts of the tangency portfolio, and (3) work out of sample in a new 93-country dataset. Evidence is "strengthened (not weakened) by the large number of observed factors." — [NBER w28432](https://www.nber.org/papers/w28432) [V]
- Harvey, Liu & Zhu ("...and the Cross-Section of Expected Returns", NBER w20592; RFS 2016): because of multiple testing over hundreds of factors, a new factor needs a t-ratio greater than 3.0. — [NBER w20592](https://www.nber.org/papers/w20592) [V; the abstract page did not show the count of factors tested or the false-discovery share]
- Chen & Zimmermann, Open Source Asset Pricing ("Open Source Cross-Sectional Asset Pricing", Critical Finance Review 2022, 27(2):207-264): free signal and portfolio-return data replicating published predictors, with a Signal Browser and a chart of original vs replicated t-stats. Download through the site, the Python package `openassetpricing`, or the R package `OpenSourceAP.DownloadR`. As of the October 2025 release, data run through 2023. — [openassetpricing.com](https://www.openassetpricing.com/) [V; the page did not state the exact predictor count or reproduction rate]

### Inferences
- HXZ and JKP mostly disagree about method (microcap handling, single-factor vs Bayesian shrinkage across clusters), not about which themes are real. For a builder, the practical rule is to start from the 13 JKP themes and the Chen-Zimmermann "clear predictor" set, use NYSE breakpoints and value weights, and require t>3 for anything new.
- HXZ's 93% failure rate for liquidity variables suggests that liquidity/trading-friction signals are the least reliable category for a large-cap book.
- The Chen-Zimmermann data are a free, ready-made signal library. Atlas's "frozen test" signal registry could benchmark against it instead of rebuilding signals from scratch.

### Gaps
- Exact Chen-Zimmermann figures (about 200+ predictors; the share whose replicated t-stats match the originals) were not on the homepage. The paper's SSRN page was not opened.
- HLZ's count of factors tested and estimated false-discovery rate could not be read (the NBER page showed only a short summary).
- I did not get a per-category JKP breakdown (which of the 13 clusters are strongest in the US since 2010).

## (b) How much does alpha decay after publication, and what about crowding?

### Takeaway
Across 97 predictors, published-anomaly returns are about 26% lower out of sample and about 58% lower after publication. Measured on modern-era data and net of trading costs, the average anomaly's expected return is close to zero (about 8 bps/month), and decay rises to about 93% after costs.

### Cited Findings
- McLean & Pontiff (JF 2016, "Does Academic Research Destroy Stock Return Predictability?"): across 97 predictors, portfolio returns are 26% lower out of sample and 58% lower post-publication, which implies about 32 percentage points (58 - 26) from publication-informed trading. The 26% is an upper bound on statistical bias. Post-publication declines are larger for predictors with higher in-sample returns, and returns are higher in high-idiosyncratic-risk, low-liquidity stocks (limits to arbitrage). — [Wiley abstract](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365) [S: Wiley returned 403; numbers from search snippet, consistent with the widely cited abstract]
- Chen & Velikov ("Zeroing In on the Expected Returns of Anomalies", JFQA 2023, 58(3)): 204 anomalies, adjusted for effective bid-ask spreads, post-publication effects, and the post-early-2000s trading-technology era. The average anomaly's net expected return is about 8 bps/month, and the strongest anomalies net only about 10-20 bps/month after data-mining adjustments. Decay is 72% when stale (pre-2005) post-publication data is excluded and 93% after trading costs. Anomaly portfolios overweight stocks with spreads about 4x the median NYSE spread and turn over about 40% of both legs each month. — [SSRN 3073681](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3073681) [S: SSRN 403]; [JFQA PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/945133D5A3ECEEAF466AEE91551FD225/S0022109022000874a.pdf/zeroing-in-on-the-expected-returns-of-anomalies.pdf) [not opened]

### Inferences
- A sensible planning assumption is that a published cross-sectional anomaly delivers at most about 30-40% of its paper return gross. In the post-2005 era, after costs, expect about zero unless it is low-turnover and implementable in liquid names.
- Decay is concentrated where in-sample returns were highest and in illiquid, high-idiosyncratic-volatility stocks. Big backtest returns in small caps are the least trustworthy.

### Gaps
- No direct crowding metric (such as short-interest or institutional-ownership growth in anomaly legs after publication) was verified this session. McLean-Pontiff report such evidence, but I could not read the PDF.
- I found no verified 2020-2026 update of post-publication decay specifically.

## (c) Which anomalies stay implementable after transaction costs, especially large vs small caps?

### Takeaway
Turnover decides survival. Anomalies with one-sided monthly turnover below about 50% (value, profitability/quality, investment, low-frequency momentum variants) generally keep significant net spreads. High-turnover ones (short-term reversal, many liquidity/microstructure signals) mostly do not. A buy/hold spread (hysteresis band) is the single most effective cost-mitigation technique.

### Cited Findings
- Novy-Marx & Velikov ("A Taxonomy of Anomalies and their Trading Costs", NBER w20721; RFS 2016): most anomalies with one-sided monthly turnover under 50% keep statistically significant net spreads. Few higher-turnover strategies do. A buy/hold spread, which lets investors keep stocks they would not actively trade into, is the single most effective simple mitigation. — [NBER w20721](https://www.nber.org/papers/w20721) [V; bps figures not on the abstract page]
- Chen & Velikov: anomaly portfolios tilt toward stocks with spreads about 4x the NYSE median and about 40% monthly turnover. These costs plus decay leave about 8 bps/month on average. — [SSRN 3073681](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3073681) [S]
- HXZ: NYSE breakpoints and value weighting, a proxy for a large-cap tradable universe, cause most anomalies to lose significance. — [NBER w23394](https://www.nber.org/papers/w23394) [V]
- McLean-Pontiff: anomaly returns are higher in high-idiosyncratic-risk, low-liquidity stocks. — [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365) [S]

### Inferences
- For a solo builder trading liquid US stocks: favor low-turnover composites (quality + value + medium-term momentum + earnings-revision blends), rebalance monthly or less often, and use hold bands (for example, enter at the top decile, exit only below the top tercile).
- Small-cap anomaly alpha mostly exists on paper because of spreads and borrow. A small account has less market impact, but bid-ask spreads and short-borrow fees still apply. Long-only implementation avoids borrow costs but loses the short leg, where much anomaly alpha sits (not verified here).

### Gaps
- Novy-Marx & Velikov's per-anomaly net spreads and cost estimates in bps were not verified (abstract page only).
- No verified source this session for the share of anomaly return coming from the short leg, or for borrow costs.

## (d1) Do PEAD, short interest, options-implied, and insider signals still work?

### Takeaway
PEAD on raw earnings surprises is essentially gone for large caps since about 2006 and only recently faded in microcaps. Options-implied signals are the most robust public-market category in recent data: the IV spread and risk-neutral skewness stay strong in 2015-2026, while the Xing-Zhang-Zhao smirk has decayed to insignificance in the AI/mega-cap regime.

### Cited Findings
- Martineau ("Rest in Peace Post-Earnings Announcement Drift", Critical Finance Review): for large stocks PEAD has not existed since 2006, and it has only recently disappeared for microcaps. Prices now fully reflect earnings surprises on the announcement date. — [SSRN 3111607](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111607) [S: SSRN 403]; [EconPapers listing](https://econpapers.repec.org/article/nowjnlcfr/104.00000122.htm) [not opened]
- Related: a Columbia paper studies why PEAD declined (role of earnings-news persistence). — [Columbia PDF](https://business.columbia.edu/sites/default/files-efs/imce-uploads/CEASA/Events%20Page/PEAD_Declined_over_time.pdf) [not opened; listed for follow-up only]
- Cremers & Weinbaum (JFQA 2010): the implied-volatility spread (ATM call IV minus ATM put IV, a measure of put-call parity deviations) is strongly positively related to future stock returns. The interpretation is that informed traders trade options first. — [Rochester "Deeper Look at the IV Spread"](https://urresearch.rochester.edu/fileDownloadForInstitutionalItem.action?itemId=35841&itemFileId=188460) [S]
- Xing, Zhang & Zhao (JFQA 2010): stocks with the steepest smirks (OTM put IV minus ATM call IV) underperform those with the flattest by 10.9% per year risk-adjusted. Predictability lasts at least 6 months, and steep-smirk firms have the worst earnings shocks the next quarter. — [Rice PDF](https://www.ruf.rice.edu/~yxing/option-skew-FINAL.pdf) [opened; the PDF rendered as binary, so the numbers are from a search snippet]
- Recent evidence (Li & Wang, arXiv 2608.26115, January 2015 to April 2026, 12.36M firm-days, 10,026 underlyings, proprietary OPRA extract): IV spread has |t| > 4 in every regime and specification. BKM risk-neutral skewness is the most robust predictor (t = -13.1 full sample) and strengthens with horizon. The smirk decays from coefficient -0.0234 (t = -5.5, 2015-2019) to -0.0060 (t = -1.5, insignificant) in the AI/mega-cap regime and reverses sign at 63 days (+0.016, t = +2.1). The 2015-2019 univariate smirk quintile spread is about -4.91% annualized. Machine learning adds value only in the AI/mega-cap regime (OOS R² +1.29% for XGBoost vs +0.07% linear). The paper does not test transaction costs and gives no large/small-cap split. — [arXiv 2608.26115](https://arxiv.org/html/2608.26115) [V; recent working paper, not peer-reviewed]

### Inferences
- Atlas should not rely on a plain SUE/PEAD drift signal for large caps. Any earnings signal should focus on information beyond the headline surprise (revisions, text, guidance).
- Options IV spread and skew are the best-supported "public but under-used" signals post-2015. The obstacle is clean end-of-day option IV data, not the economics. An options-chain API (Atlas already has Alpaca option snapshot and chain tools) could be used to compute ATM call-put IV spreads daily, but the history needed for a frozen backtest would have to be built forward or bought.

### Gaps
- Short interest / days-to-cover: no source verified this session (for example Boehmer, Jones & Zhang; Hong et al. days-to-cover; Rapach, Ringgenberg & Zhou 2016). Still to do: post-2010 return spreads and FINRA bi-monthly data lags.
- Insider trading (SEC Form 4): no source verified (for example Cohen, Malloy & Pomorski 2012 "Decoding Inside Information", opportunistic vs routine trades). Still to do: post-2010 alpha after costs.
- Analyst revisions: no verified post-2010 net-of-cost evidence gathered.
- I did not find whether IV-spread alpha survives option-implied bid-ask noise and stock trading costs in large caps.

## (d2) What does the alternative data evidence show, how fast does it decay, and what costs fit a small budget?

### Takeaway
Paid alternative data (such as satellite parking-lot counts) did contain real information: about 4-5% in the three days around retailer earnings in 2011-2017. But it stayed concentrated among well-funded funds and needs expensive skills, and no published decay estimate was verified here. For a small budget, the realistic options are free proxies (Google Trends, Wikipedia views, Form 4, FINRA short interest, options chains), whose evidence I could not verify this session.

### Cited Findings
- Katona, Painter, Patatoukas & Zeng ("On the Capital Market Consequences of Alternative Data: Evidence from Outer Space", JFQA): 4.8 million parking-lot images at 67,000 stores of 44 major US retailers (Walmart, Target, Costco, Whole Foods...), 2011-2017. Satellite data predicts quarterly sales and yields 4-5% in the three days around earnings announcements. The data is expensive and needs costly analysts, so the gain goes to hedge funds and large institutions, with retail often on the other side. There is no overall improvement in market price discovery, and there is evidence of informed short-selling before reports at covered retailers. — [Berkeley Haas News](https://newsroom.haas.berkeley.edu/how-hedge-funds-use-satellite-images-to-beat-wall-street-and-main-street/) [V]; [Semantic Scholar record](https://www.semanticscholar.org/paper/On-the-Capital-Market-Consequences-of-Alternative-Katona-Painter/744b68782d90088c929b97ba82bedf552468b889) [not opened]
- Satellite imagery of container ports has been reported to predict world stock returns. — [Humanities & Social Sciences Communications (Nature)](https://www.nature.com/articles/s41599-023-01891-9) [not opened; snippet only]
- Free/cheap options-chain data underlies the strongest recent public signals above (IV spread, skew). The academic version used a proprietary OPRA extract. — [arXiv 2608.26115](https://arxiv.org/html/2608.26115) [V]

### Inferences
- Paid alternative data (satellite, card panels, app downloads) is out of reach for a solo budget, and the edge sits around earnings events, where institutions with the same data are already active. Once vendors sell a dataset widely, the McLean-Pontiff logic of decay after the information spreads should apply, though that is not verified for alternative data.
- The best low-cost options are, in order: (1) daily options-implied IV spread and skew from a broker API, (2) Chen-Zimmermann free signal library, (3) SEC EDGAR Form 4 (free) for opportunistic insider buys, (4) FINRA short interest (free, about twice a month, lagged), and (5) Google Trends and Wikipedia page views as attention proxies. Items 3-5 lack verified recent evidence (see Gaps).

### Gaps
- Card transactions (for example Second Measure / Earnest panels), web traffic, app downloads, and job postings: no peer-reviewed numbers or decay estimates verified this session.
- Google Trends (Da, Engelberg & Gao 2011, "In Search of Attention") and Wikipedia page views: not verified this session; also needed is post-2010 out-of-sample evidence.
- Vendor pricing for alternative datasets: not found.
- I found no verified quantitative estimate of alpha decay after alternative data became widely sold.

## (e) What IC and Sharpe are realistic for a solo builder (fundamental law)?

### Takeaway
I verified no source this session. Using the verified decay and cost numbers above, a public-data cross-sectional book should be planned around a small IC and a net Sharpe well below backtest values.

### Cited Findings
- Net of costs and post-publication decay, the average anomaly earns about 8 bps/month and the best about 10-20 bps/month. — [SSRN 3073681](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3073681) [S]
- Recent ML out-of-sample R² for option-signal return prediction is about 0.07% (linear) to 1.29% (XGBoost, one regime). This shows how small achievable predictive power is even for strong signals. — [arXiv 2608.26115](https://arxiv.org/html/2608.26115) [V]

### Inferences
- With IR = IC x sqrt(breadth) (Grinold's fundamental law; not source-verified here): an IC of 0.02-0.03 on about 500 stocks rebalanced monthly gives roughly 0.02 x sqrt(6000), about 1.5 gross on paper. The transfer coefficient (long-only and position constraints), costs, and decay would plausibly cut that by half or more. This is illustrative arithmetic, not a sourced estimate.
- Atlas's existing frozen-holdout registry is the right defense against the 26-58% decay. Reported holdout ICs should be haircut further for costs before ranking picks.

### Gaps
- No source verified for Grinold-Kahn fundamental law, typical practitioner ICs (often cited as 0.02-0.05), or transfer-coefficient values. A follow-up should cite Grinold (1989) or Clarke, de Silva & Thorley (2002).
