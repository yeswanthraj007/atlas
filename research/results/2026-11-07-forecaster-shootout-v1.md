# Result: forecaster shootout v1 (ARIMA-GARCH, LSTM, MarketGPT, Chronos, TimesFM)

Pre-registration: research/preregistered/2026-11-07-forecaster-shootout-v1.md (with
Amendments 1 and 2). Trials: #70-74 (first run), #75-76 (ensembles, first run), #77
(ARIMA-GARCH after the fix), #78-79 (ensembles after the fix).
Frozen test, S&P 500, every 20th trading day from 2024-01-02, 35 dates, 20-day horizon.

## Disclosure: a bug found after the first results
In the first run, stocks that began trading after the fitting cut-off (SOLV and GEV, spun off in
2024, and others) had their missing history zero-padded, so ARIMA-GARCH was fitted to a
flat line and forecast near-zero volatility. That pushed its mean QLIKE to 4.98 (trimmed of
the worst 1%: 0.26). The fix fits only on real returns and needs 500 real days; 9 stocks
drop out of every model's evaluation set. The other models' saved predictions were reused
unchanged. The tables below are the corrected ones; the first-run trials stay in the log.

## Task R: 20-day return ranking (15,140 stock-dates; ridge baseline IC -0.0247)
| Model | IC | t | Accuracy | t vs ridge |
|---|---|---|---|---|
| Chronos | +0.0239 | 0.77 | 51.1% | 0.83 |
| MarketGPT | +0.0199 | 0.86 | 50.8% | 1.30 |
| TimesFM | +0.0188 | 0.62 | 51.1% | 0.73 |
| LSTM | +0.0063 | 0.20 | 50.4% | 1.48 |
| ARIMA-GARCH | +0.0027 | 0.12 | 50.0% | 0.68 |

No model is useful for returns (t >= 3 against ridge needed). All ICs are below 0.03.

## Task V: 20-day volatility (QLIKE, lower is better; HAR baseline 0.3522)
| Model | QLIKE | DM t vs HAR | Above-median accuracy |
|---|---|---|---|
| ARIMA-GARCH | 0.3068 | +1.87 | 70.6% |
| LSTM | 0.3600 | -0.40 | 74.0% |
| MarketGPT | 0.3654 | -0.46 | 75.4% |
| TimesFM | 0.3876 | -2.74 | 71.1% |
| Chronos | 0.4008 | -3.86 | 70.7% |
| HAR (baseline) | 0.3522 | — | 71.9% |
| naive (baseline) | 0.4650 | -4.85 | 69.2% |

ARIMA-GARCH has the lowest QLIKE and MarketGPT the best ranking accuracy, but neither
beats HAR with t >= 3. The two pretrained foundation models are significantly WORSE than HAR.

## Ranking (mean of the two ranks)
MarketGPT 2.5; ARIMA-GARCH, LSTM and Chronos 3.0; TimesFM 3.5. The differences are small.
Best overall by the pre-registered rule: MarketGPT. Useful for either task: none.

## Ensemble stage (Amendment 2): members and calibration chosen on 2024, scored on 2025-26
(22 dates, 9,633 stock-dates)
- Returns: top-3 (Chronos, TimesFM, ARIMA-GARCH) IC -0.009; all five IC +0.019 (t 0.54).
  Neither beats the best single model (paired t -0.62 and +0.58).
- Volatility: top-3 (LSTM, MarketGPT, ARIMA-GARCH) QLIKE 0.380 against a calibrated HAR 0.402
  (DM t +2.02: short of 3.0); all five 0.394 (t +0.64). Neither beats the best single model,
  LSTM at 0.345 (paired t -0.75 and -0.93).
- Calibrating ARIMA-GARCH on 2024 made it worse in 2025-26 (0.497 against 0.307 uncalibrated
  over the whole test), so its volatility relationship moved.

## What we learned
1. For stock returns, no model family helps: not classical, not recurrent, not transformer,
   not pretrained.
2. For volatility, a classical GARCH model is as good as the neural networks, and the pretrained
   foundation models are the weakest.
3. Combining models did not beat picking the best one here. The members' errors are too similar.
4. The registry's validated volatility model (76% above-median accuracy, t = 33) stays the
   production choice.
