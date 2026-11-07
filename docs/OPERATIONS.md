# Operations: scheduled jobs on this Mac

## Daily option implied-volatility recorder
Records a compact summary of every S&P 1500 stock's option chain (implied volatility, ATM
straddle price, skew, per expiry) to `data/options_iv/<date>.parquet`. Alpaca serves only the
current chain, so this builds our own history.

- Schedule: Monday to Friday at 14:00 local time (5 PM Eastern), via launchd.
- Definition: `~/Library/LaunchAgents/local.atlas.record-option-iv.plist`
- Log: `data/logs/record_option_iv.log`
- Resumable and idempotent: re-running a day skips stocks already recorded; on market
  holidays it finds the previous day's file complete and does nothing.
- The Mac must be awake (plugged in, lid open) at that time. If it was asleep, macOS runs the
  job when it wakes.

Commands
```bash
launchctl print gui/$(id -u)/local.atlas.record-option-iv        # status and run count
launchctl kickstart -k gui/$(id -u)/local.atlas.record-option-iv # run it now
launchctl bootout gui/$(id -u)/local.atlas.record-option-iv      # stop and remove the schedule
```
Data caveat: Alpaca's free plan serves the "indicative" option feed, not live exchange quotes;
some implied volatilities are unreliable (for example very short-dated at-the-money options).
