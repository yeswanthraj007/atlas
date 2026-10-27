# Result: MarketGPT v3 — rejected

Pre-registration: research/preregistered/2026-10-26-market-gpt-v3.md, including Amendments 1–3.
Trials: #64 (ensemble) and #65 (price-only ablation); ridge baseline logged.

## Frozen test, 2024-01 to 2026-09
35 monthly dates, S&P 1500, 50,997 stock-months.

| Model | Test IC | t | Net decile spread/month (tau 0.5) |
|---|---|---|---|
| Ensemble, pre-registered averaging (per-seed z-score) — **verdict** | +0.0077 | 0.47 | +0.72% |
| Ensemble, main-script averaging (raw mean, then z-score) | +0.0102 | 0.61 | +0.68% |
| Price-only, seed 0 | +0.0121 | 0.59 | — |
| News, seed 0 | -0.0001 | -0.01 | — |
| Ridge baseline | +0.0004 | 0.01 | — |

Success criteria:
1. t >= 3.0: **FAIL**.
2. Net decile spread > 0: pass.
3. Beats the ridge baseline: pass.

Verdict: **rejected**. The t-statistic is below both the 3.0 promotion bar and the 2.0
watchlist bar.

News ablation: news minus price-only, seed 0, IC -0.012 (paired t -0.90). **News adds no
value.**

## Descriptive metrics (Amendment 3; reported only, not part of the verdict)
- Monthly win rate, net top-minus-bottom decile: 19/35 = 54% (binomial p = 0.37).
- Beats-median accuracy: 50.3% for all stocks. For the most confident calls: 51.1% for the
  top/bottom 5%, 51.2% for the top/bottom 10%.
- Long-only top decile beat the equal-weight average in 19/35 months (54%).
- IC by year: 2024 +0.008, 2025 +0.009, 2026 +0.005.
- Single-seed test ICs: -0.000, +0.015, +0.005.

## Disclosure
The main script averaged the seeds' raw scores, not the per-seed z-scores the
pre-registration specifies. The pre-registered version (scripts/market_gpt_v3_eval.py,
written and committed before the test) gives the verdict. Both versions reject.

## What we learned
- Validation ICs (+0.02 to +0.075) did not carry into the test (+0.008). There are two
  reasons:
  - Selecting the best epoch on a noisy validation year inflates validation scores.
  - The edge decays over time.
- Hiding the date-fingerprinting market input was a real engineering fix: training
  correlation fell from 0.38 to about 0.1, and validation became consistent. But the edge
  left after the fix is too small to measure in 35 months.
- More data (16x) and better training did not produce a usable direction signal for
  20-day cross-sectional returns in US equities 2024-2026. This matches every earlier trial
  (#1-#63).
