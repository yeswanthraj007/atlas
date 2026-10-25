# Research process

How a new trading signal gets from an idea to the dashboard. Modelled on how Man Group
gates its AI-generated research: an AI may propose ideas and write their code, but every
idea, human or AI, passes the same fixed gates, and every attempt is logged.

## The loop

```
1. Propose     uv run atlas research new <name>        (scaffolds src/atlas/candidates/<name>.py)
2. Write       hypothesis + economic rationale FIRST, then compute()
3. Evaluate    uv run atlas research run <name>
4. Review      uv run atlas research log
```

## The gates (atlas/lab/evaluate.py)

| Gate | Test | Why |
| --- | --- | --- |
| 1. Look-ahead | Recompute on data cut off at earlier dates; values must be identical | A signal that changes when the future is removed was using the future |
| 2. Development | 2017–2024: IC t-stat ≥ 3 and the same sign in ≥ 60% of years | t ≥ 3 is the Harvey–Liu–Zhu bar for new factors; consistency guards against one lucky year |
| 3. Frozen holdout | 2025 onward, opened **only** after gate 2 passes: same sign, t ≥ 2 | Each look at the holdout makes it less unseen; failed ideas never look |
| 4. Deflated Sharpe | Long-short Sharpe still significant (P ≥ 0.95) after counting **every** trial in the log | Trying many ideas and keeping the best is luck unless all attempts are counted |

Passing all four writes a `direction` entry to the signal registry; only then can the
signal rank picks on the dashboard.

## Rules

- **Rationale before results.** A candidate without a written hypothesis and an economic
  reason is refused. "The backtest looked good" is not a reason.
- **The trial log is append-only** (`research/trials.jsonl`, committed to git). Failures
  stay. A fixed or tweaked idea is a new trial, never an edit of an old one.
- **Never edit a candidate after its holdout opened.** Change it under a new name.
- **Disclose weaknesses** in `SPEC.notes`, e.g. a holdout already seen in exploration.
- **Language models propose and explain, they do not decide.** No LLM output ranks stocks
  directly; only signals that passed the gates do.
- **Text models used on history must predate the test period** (pinned revisions), so they
  cannot carry memorised knowledge of the outcomes.

## Where things live

| Path | What |
| --- | --- |
| `src/atlas/candidates/` | Candidate signals, one module each |
| `src/atlas/lab/` | Trial log and evaluator |
| `research/trials.jsonl` | Every trial ever run |
| `data/signal_registry.json` | Signals that passed (and their test results) |
| `scripts/holdout_*.py` | Frozen tests for signals that need their own training loop (news, earnings text, volatility) |
