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

## Frontier-LLM forward test (pre-registered 2026-11-07)

Rules: `research/preregistered/2026-11-07-llm-forward-test-v1.md`. Nothing here is scheduled yet.

- Needs `ANTHROPIC_API_KEY` in `.env` (typed by the owner; never committed). Without it the runner exits.
- `uv run python scripts/llm_forward_run.py` pulls new S&P 1500 earnings releases, predicts those whose
  reaction session has not started, appends to `data/llm_forward/predictions.jsonl` and writes the day's
  SHA-256 digest to `research/forward/hashes/<date>.txt`. Commit and push the digest the same day: the git
  timestamp is the external record. Start date 2026-11-08; monthly spend cap $75.
- Plumbing check without cost: `--dry-run --ignore-start --pretend-fresh --since-days 6 --limit 40`.
- Intended cadence once approved: 08:45 ET and 16:45 ET on weekdays (before the open, after the close).
- `uv run python scripts/llm_forward_score.py` refreshes nothing; refresh S&P 1500 bars first. It scores only
  at 600, 1,500 and 3,000 complete 20-day outcomes (state in `data/llm_forward/checkpoints.json`) and logs one
  trial per checkpoint. `--dry-run --force` checks the plumbing.
- Never edit the predictions file or the digests, never change `prompt.py` (its SHA is stored per row).

Scheduled (approved 2026-11-09): `~/Library/LaunchAgents/local.atlas.llm-forward.plist` runs
`scripts/run_llm_forward.sh` Mon-Fri at 05:45 and 13:45 Pacific (08:45 and 16:45 ET). The wrapper predicts,
then commits and pushes only `research/forward/hashes`. Log: `data/logs/llm_forward.log`. Stop it with
`launchctl bootout gui/$(id -u)/local.atlas.llm-forward`. The Mac must be plugged in and awake.
