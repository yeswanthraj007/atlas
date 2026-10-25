# Practical Constraints for a Solo Retail Developer Running Automated US Equities Trading via Alpaca (as of Oct 2026)

Research note: the web tools hit a session limit partway through. Everything under "Cited Findings" was opened and checked during this session. Topics I could not reach (academic studies, latency, spreads, tax rules, settlement and good-faith rules) are listed under "Gaps" with leads only. They are not stated as fact.

## 1. PDT rule (FINRA Rule 4210): current status

### Takeaway
**The big change: the Pattern Day Trader rule no longer exists in its old form.** The SEC approved FINRA's amendments to Rule 4210 on April 14, 2026, and they took effect June 4, 2026. They remove the "pattern day trader" definition, the 4-day-trades-in-5-business-days count and the $25,000 minimum equity. A risk-based "intraday margin" standard replaces them. Brokers may phase it in until October 20, 2027. Alpaca switched on June 4, 2026. The $2,000 minimum equity needed to use margin still applies.

### Cited Findings
- SEC approval date: April 14, 2026. Securities Exchange Act Rel. No. 105226, published at 91 Fed. Reg. 20731 (April 17, 2026). Effective June 4, 2026. Firms that need system upgrades may phase in until October 20, 2027. FINRA Regulatory Notice 26-10 was issued April 20, 2026. — [WilmerHale client alert, Apr 23 2026](https://www.wilmerhale.com/en/insights/client-alerts/20260423-sec-approves-amendments-to-finra-rule-4210-replacing-day-trading-margin-requirements-with-a-modernized-intraday-margin-standard)
- FINRA describes the change as new intraday margin standards that replace the old day-trading margin requirements "in their entirety". That includes the day-trade count used to designate a "pattern day trader" and the $25,000 minimum equity. The effective date is June 4, 2026, 45 days after the Notice was published, and the phase-in runs to October 20, 2027. — [FINRA Regulatory Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10) (from search snippet; the page itself was not opened)
- How the new framework works (per WilmerHale's summary of the rule):
  - Firms must calculate an intraday margin deficit for each customer margin account. They can either monitor in real time and block trades that would create a deficit, or calculate at end of day and issue prompt margin calls.
  - Customers must cover a deficit "as promptly as possible", by depositing funds or liquidating positions.
  - A deficit stays outstanding until it is covered, or until the close of business on the 15th business day after it arose.
  - Customers who repeatedly fail to cover deficits promptly have their margin account frozen from extensions of credit for 90 days.
  - Good-faith accounts and portfolio-margin accounts are exempt. Maintenance margin rules (4210(c)) are unchanged.
  — [WilmerHale](https://www.wilmerhale.com/en/insights/client-alerts/20260423-sec-approves-amendments-to-finra-rule-4210-replacing-day-trading-margin-requirements-with-a-modernized-intraday-margin-standard)
- FINRA's investor page says the $2,000 minimum equity for leveraged trading still applies, and that you can trade in a margin account below $2,000 if you don't use margin. It also warns: "your firm might continue operating under the old day trading margin requirements during the transition". — [FINRA Investor Insights: Understanding the New Intraday Margin Requirements](https://www.finra.org/investors/insights/intraday-margin-requirements)
- Alpaca's blog (posted April 27, 2026, updated June 4, 2026):
  - Alpaca switched to the new framework on June 4, 2026.
  - Intraday leverage is 4x and overnight leverage is 2x. The equity needed for 4x intraday buying power dropped from $25,000 to $2,000.
  - These API fields were to be removed by July 6, 2026: `pattern_day_trader`, `daytrade_count`, `last_daytrade_count`, `daytrading_buying_power`, `last_daytrading_buying_power`. `buying_power` now reflects intraday availability and updates through the day.
  — [Alpaca blog: FINRA Retires the PDT Rule](https://alpaca.markets/blog/finra-retires-the-pdt-rule-introducing-alpacas-new-intraday-margin-framework/)
- Alpaca's docs page has more on deficits. An unmet deficit after five business days triggers a 90-day restriction on new debits and new short positions; closing existing shorts is still allowed. De minimis exceptions apply to deficits under $1,000 or 5% of equity. **Conflict:** the same page mentions a "12-month interim transition period", which contradicts the 18-month phase-in (to Oct 20, 2027) that FINRA and WilmerHale report. Treat FINRA's figure as authoritative. — [Alpaca Docs: Understanding FINRA's New Intraday Margin Rule and the End of PDT](https://docs.alpaca.markets/docs/understanding-finras-new-intraday-margin-rule-and-the-end-of-pdt)
- Alpaca's 2026 data plans and account docs do not reproduce the old PDT definition: 4+ day trades in 5 business days, where day trades are more than 6% of total trades, in a margin account. That definition is now historical in any case.

### Inferences
- For a solo developer at Alpaca, the $25k barrier is gone. A margin account with at least $2,000 can day trade freely, limited only by real-time intraday buying power (4x). Code that reads `daytrade_count` or `pattern_day_trader` breaks after July 6, 2026 and must move to `buying_power`.
- A new failure mode replaces the PDT lockout. Intraday margin deficits that go uncovered can trigger a 90-day credit freeze. A bot must check buying power before each order and must not rely on the broker to reject bad orders.
- Cash accounts are still an option for those who want no leverage. T+1 settlement applies (see Gaps), and good-faith and free-riding rules still bind there, because the PDT rewrite concerned margin accounts only.
- Before relying on the new rules at any other broker, check whether that broker has adopted them. They may keep legacy PDT until Oct 2027.

### Gaps
- I could not open the Federal Register PDF (govinfo FR-2026-04-17/2026-07485) or the PDF text of Regulatory Notice 26-10 for exact wording.
- **T+1 settlement and cash-account good-faith violations.** The move to T+1 took effect May 28, 2024 (SEC Rule 15c6-1 amendments). Good-faith violations are a Regulation T cash-account concept. Neither was verified this session, so cite sec.gov's T+1 pages before publishing.
- I did not find whether Alpaca paper accounts now simulate intraday margin deficits.

## 2. Alpaca specifics

### Takeaway
Alpaca has two data plans: Free (IEX only, 30 WebSocket symbols, 200 API calls/min) and Algo Trader Plus at $99/month (all US exchanges/SIP, unlimited symbols and calls). Both offer 7+ years of history. The full suite of order types is supported. Extended hours run 4:00am–8:00pm ET plus an overnight session from 8:00pm to 4:00am, Sunday through Friday. Paper trading fills against the NBBO and ignores liquidity, market impact and queue position. Alpaca earns payment for order flow from wholesalers.

### Cited Findings
- **Data plans:**
  - Free: $0, IEX feed only. Real-time via WebSocket, but REST API data is delayed 15 minutes. 200 API calls/min, max 30 WebSocket symbols, indicative options quotes.
  - Algo Trader Plus: $99/month, "All US Exchanges". Real-time on both WebSocket and API, unlimited API calls, unlimited symbols, real-time options.
  - Both plans: 7+ years of history, extended hours data, crypto, bars and snapshots.
  — [Alpaca Market Data pricing](https://alpaca.markets/data)
- **Order types:** market, limit, stop, stop-limit, trailing stop, bracket, OCO and OTO.
  - Time in force: DAY, GTC (auto-canceled after 90 days), OPG (opening auction), CLS (closing auction), IOC and FOK.
  - Extended hours accept limit orders only, with DAY or GTC time in force. Bracket orders cannot use extended hours.
  - Sessions: overnight 8:00pm–4:00am ET (Sun–Fri), pre-market 4:00–9:30am, after-hours 4:00–8:00pm.
  - Fractional shares: the docs summary listed limit, stop and stop-limit, DAY only. Notional orders cannot be replaced.
  - Prices of $1.00 or more allow at most 2 decimals; prices under $1.00 allow 4.
  — [Alpaca Docs: Orders at Alpaca](https://docs.alpaca.markets/docs/orders-at-alpaca)
- **Paper trading:**
  - Orders are matched against the current NBBO. Order size is not checked against NBBO size, so fills can be "much larger than the actual available liquidity".
  - Partial fills of random size happen 10% of the time.
  - Not simulated: market impact, information leakage, slippage from latency, queue position, price improvement, dividends and regulatory fees. Borrow fees are listed as "coming soon".
  - New accounts start with $100k by default. Accounts can be created and deleted but not reset.
  — [Alpaca Docs: Paper Trading](https://docs.alpaca.markets/docs/paper-trading)
- **Payment for order flow:** Alpaca Securities receives PFOF from Virtu Americas, Citadel Execution Services and Jane Street for equity order flow, and the payment varies with the stock's price. Quarterly SEC Rule 606 reports are at alpaca.markets/disclosures. — [Alpaca 606 report (2024Q3)](https://files.alpaca.markets/disclosures/library/SEC+606a1+-+2024Q3.pdf); [Alpaca Learn: Inside Payment for Order Flow](https://alpaca.markets/learn/love-it-or-hate-it-inside-payment-for-order-flow-and-commission-free-trading-apps) (both from search snippets; the PDFs were not text-extractable)

### Inferences
- On the free plan, IEX carries only a small slice of consolidated volume, so its quotes and trades can differ from the NBBO. Intraday signals built on spreads, quotes or volume need SIP ($99/month). The 30-symbol WebSocket cap also limits universe scanning.
- Paper P&L overstates live results, especially for market orders, larger size, illiquid names and strategies that depend on fill priority. Haircut paper results by a modeled spread and impact cost.
- Under PFOF, marketable orders go to wholesalers and usually get some price improvement, but routing is not under the user's control. Limit orders behave differently from direct exchange access.

### Gaps
- The Trading API rate limit (commonly cited as 200 requests/min per account) was not verified.
- The news WebSocket stream (Benzinga-sourced) was not verified this session.
- Alpaca does not appear to publish API or order latency figures. None was found, and the search was cut short.
- The fractional market-order rules need re-checking against the live docs. The fetch summary may have left out market orders.

## 3. Transaction costs for retail-sized orders

### Takeaway
Explicit costs at Alpaca are close to zero: no commission and small regulatory fees on sales only. The SEC Section 31 fee has been $20.60 per $1M of sales (0.206 bps) since April 4, 2026; it was $0 from Oct 2025 to April 3, 2026. The FINRA TAF is about $0.000195/share in 2026, capped at $9.79. Implicit costs (spread, slippage, impact) dominate.

### Cited Findings
- **SEC Section 31 fee:** $20.60 per million dollars of covered sales on or after April 4, 2026. It was $0.00 per million through April 3, 2026. It stays in effect until 60 days after the FY2027 appropriation is enacted. — [SEC Fee Rate Advisory 2026-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2)
- **FINRA TAF:** $0.000195/share on sales of covered equity, with a $9.79 per-trade cap, effective Jan 1, 2026 (up from $0.000166 and $8.30) under the fee schedule in SR-FINRA-2024-019, which steps up through 2029. — [FINRA Fee Adjustment Schedule](https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule) (search snippet). **Conflict:** the FINRA rulebook page I opened still shows $0.000166 and $8.30 — [FINRA By-Laws Schedule A, Section 1](https://www.finra.org/rules-guidance/rulebooks/corporate-organization/section-1-member-regulatory-fees). This is probably a stale rulebook page. Check the fee schedule page before relying on either figure.

### Inferences
- Worked example: selling 100 shares at $200 ($20k) costs about $0.41 in SEC fee plus $0.02 in TAF, roughly 0.2 bps. That is negligible next to a 1–2 bps spread in mega-caps.
- Round-trip cost is roughly the full quoted spread for market orders, minus any price improvement, plus slippage from signal latency. Even a 1-cent spread on a $50 stock is 2 bps per round trip. Intraday signal edges of a few bps per trade can be wiped out completely. Any backtest should charge at least half the spread plus a slippage buffer per side.

### Gaps
- I verified no published bps figures for large-cap quoted or effective spreads in 2026. Candidate sources are SEC Rule 605 reports, the SEC's "Disclosure of Order Execution Information" adopting release (2024), and academic papers on retail execution quality. The tool limit stopped me from checking any of them.
- No source was verified on how typical intraday signal edges in bps compare with costs.

## 4. Latency realities

### Takeaway
I could not verify this session. Alpaca publishes no latency figures that I found.

### Cited Findings
- None verified.

### Inferences (not sourced)
- A retail stack sends REST/WebSocket calls over the public internet to a broker that routes to wholesalers. Round trips are in the tens of milliseconds at best; a VM in AWS us-east-1, near the NJ data centers, does better than home internet. Strategies whose edge decays in microseconds to milliseconds (market making, latency arbitrage) are not reachable. Strategies on minute-to-day horizons are.

### Gaps
- Alpaca's server location and any published order-ack latency.
- Measured home vs us-east-1 round-trip times.
- Academic evidence on alpha-decay horizons.

## 5. Operational best practice, tax, and retail day-trader outcomes

### Takeaway
Only partly verified. The useful verified point is the paper-trading limitations above, which show why paper results overstate live results.

### Cited Findings
- Paper trading does not model impact, queue position or latency slippage, and lets fills exceed displayed liquidity. Paper performance is therefore an upper bound. — [Alpaca Docs: Paper Trading](https://docs.alpaca.markets/docs/paper-trading)
- Repeatedly failing to cover intraday margin deficits leads to a 90-day credit freeze. This should be handled by a kill switch and buying-power check. — [WilmerHale](https://www.wilmerhale.com/en/insights/client-alerts/20260423-sec-approves-amendments-to-finra-rule-4210-replacing-day-trading-margin-requirements-with-a-modernized-intraday-margin-standard)

### Inferences (operational design, not sourced)
- **Before each order:** check `buying_power`, a per-symbol position cap and a daily loss limit.
- **Kill switch:** flatten all positions and cancel all orders (Alpaca has endpoints for both).
- **Reconcile:** after every fill and at end of day, compare local state with broker positions and orders.
- **Monitoring:** alert when data goes stale, for example when the WebSocket heartbeat lapses.
- **Paper trading:** run long enough to cover several market regimes, then go live at small size and measure live slippage against paper.

### Gaps (needs verified sources before inclusion)
- **Barber, Lee, Liu & Odean (Taiwan day traders):** commonly cited for the finding that fewer than 1% of day traders are predictably profitable after fees. I could not open the source; cite the J. Financial Markets 2014 paper "The cross-section of speculator skill" or the SSRN version after checking.
- **Chague, De-Losso & Giovannetti, "Day Trading for a Living?" (Brazil, 2019):** commonly cited for the finding that about 97% of people who day traded more than 300 days lost money, and about 1% earned more than minimum wage. Not verified; check the figures on SSRN.
- **Tax:** IRS Publication 550 covers the wash sale rule (30 days before or after a sale) and short-term gains taxed as ordinary income; trader tax status / Section 475(f) is relevant too. Not verified this session, and this is not tax advice.
- I found no authoritative recommended length for paper trading.
