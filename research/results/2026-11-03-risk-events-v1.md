# Result: risk and event predictors v1

Pre-registration: research/preregistered/2026-11-03-risk-events-v1.md.
Trials #67 (E1), #68 (E2), #69 (D1). Frozen test: 2024-01 to 2026-09, S&P 1500, by month
(33 months). Models: gradient-boosted trees, trained to 2023.

| Target | Test rows | Accuracy | AUC | Best naive baseline | Margin (paired monthly t) | Most confident 20% each side |
|---|---|---|---|---|---|---|
| E1 earnings move, vs that quarter's median | 16,653 | 63.4% | 0.691 | 62.4% (own median move) | +1.0% (t 1.07) | 72.3% |
| E2 earnings move, vs the stock's own history | 16,422 | 58.6% | 0.621 | 51.3% (5d/63d vol) | **+7.3% (t 6.70)** | 64.8% |
| D1 worst 21-day drawdown, vs that date's median | 48,334 | 58.4% | 0.614 | 58.3% (63d vol rank) | +0.0% (t 0.20) | 64.2% |

Criteria: A (accuracy >= 65%) failed for all three. B (beats the best baseline, paired t >= 3)
held only for E2. Promotion needs B and accuracy >= 60%: E2 reached 58.6%, so **none is
promoted**.

Wikipedia attention adds nothing measurable: +0.17% (t 0.84), -0.04% (t 0.06), +0.29% (t 1.48).

## Reading
- E1's accuracy comes mostly from the stock's own typical earnings move, which a one-line
  baseline already captures (62.4%). The model adds a point.
- E2 is the real finding: whether a release is a bigger surprise in size than the stock's
  usual one is predictable (+7.3 points over the best baseline, t 6.7). It misses the 60% floor.
- D1: the 63-day volatility ranking already captures what the model finds.
- "Most confident" figures are a selected subset and were not compared with the baselines on
  the same subset.

## Disclosure
Six trials were pre-registered (three targets and three ablations). Three were logged; the
ablation numbers are inside each target's metrics.
