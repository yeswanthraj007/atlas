# Atlas — engineering guide

Institutional-grade portfolio, risk and trading platform (Aladdin-style), built by one
person. Read `docs/ARCHITECTURE.md` for the design and the roadmap.

## Commands

- `uv sync` — install dependencies
- `uv run pytest` — run the full test suite (must pass before every commit)
- `uv run ruff check src tests && uv run ruff format src tests` — lint and format

## Non-negotiable rules

1. **One source of truth.** Positions are always derived from the append-only `Ledger`.
   Never store or edit a position directly. Fix mistakes with a cancelling transaction.
2. **Money is Decimal.** Quantities, prices and cash in the ledger and OMS are `Decimal`.
   Risk and analytics use float64 numpy; convert only at the boundary (`to_decimal`).
3. **Every order goes through `OrderManager.submit`.** Strategies, the AI layer and the UI
   never call a broker. Pre-trade rules fail closed: a rule that raises rejects the order.
4. **Risk limits exempt only position-reducing orders**, so the kill switch never traps a
   position.
5. **No look-ahead.** In backtests, a decision on bar t fills at bar t+1 at the earliest.
   Any new data or strategy code needs a test that proves this.
6. **Every model has a test against a known answer** (closed form, textbook value, or
   simulation convergence). VaR models also get Kupiec/Christoffersen backtests.
7. **Live trading is opt-in.** Paper trading is the default. Credentials live in `.env`
   (never committed) and are read from the environment only.

## Layout

`src/atlas/<module>` — refdata, ledger, marketdata, risk, pricing, optimizer, oms,
execution, strategy, backtest, api. Tests mirror modules in `tests/`.
