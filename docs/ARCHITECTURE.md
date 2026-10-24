# Architecture

## Design decision: a modular monolith, not microservices

Aladdin runs thousands of services because thousands of engineers work on it. For one
person, running Kafka plus five services would be the real compromise: more things fail,
and the time goes on operations instead of correctness. Atlas is a single Python
application with Aladdin's internal boundaries. Each module has one public interface
and its own tests, so any module can become a separate service later without a rewrite.
Compute-heavy work (Monte Carlo, scenario grids) scales out through Ray when needed.

```
            React dashboard  ──>  FastAPI (auth, audit)
                                      │
   ┌──────────────┬──────────────┬────┴─────────┬──────────────┬──────────────┐
 marketdata     refdata         risk         optimizer      strategy        AI layer
   │              │              │              │              │              │
   └──────────────┴──────┬───────┴──────────────┴──────────────┴──────────────┘
                         │
              OMS (pre-trade risk, kill switch, audit)  ──>  broker adapters
                         │                                   (sim, Alpaca, IBKR)
               append-only ledger  ──>  position book (IBOR)
```

## Data flow of a trade

1. A strategy (or a person, or the AI layer) produces an `OrderRequest` or target weights.
2. `OrderManager.submit` builds a `PreTradeContext` from the live book and prices.
3. `PreTradeRiskEngine` runs every rule; any violation rejects the order and logs why.
4. Accepted orders go to the broker adapter. Fills come back through `on_fill`.
5. Each fill becomes a `Transaction` in the ledger; the position book updates from it.
6. Every step is recorded in the audit log.

## Roadmap (solo build)

| Milestone | Scope | Exit gate |
| --- | --- | --- |
| **M1 Foundation** (done) | Security master, bitemporal ledger, VaR/ES (historical, parametric, Monte Carlo), covariance, factor model, stress, VaR backtests, BSM Greeks, optimizer, OMS with risk rules, simulated broker, backtester | 40 tests passing |
| **M2 Market data** | Alpaca historical bars (stocks, ETFs, options), Parquet store, split/dividend adjustment, data quality checks, Fama-French factors | 10 years of daily data for 500+ names, validated |
| **M3 Paper trading** | Alpaca broker adapter, broker-vs-ledger reconciliation, streaming quotes, scheduler, Postgres persistence for ledger, orders and audit | 30 days of paper trading with zero reconciliation breaks |
| **M4 Platform UI** | FastAPI, React dashboard: positions, P&L, VaR with backtest, stress tests, factor exposures, order blotter, kill switch | Every number on screen traces to a ledger row or a risk run |
| **M5 Strategies** | Multi-factor (momentum, value, quality, low vol) with the optimizer; intraday engine on minute bars; options (vol surface, QuantLib, Greeks limits); walk-forward validation, deflated Sharpe ratio | Each strategy passes out-of-sample tests before paper trading |
| **M6 AI layer** | Claude agent reads news and filings, proposes trades with written rationale; the risk engine gates them; a human approves | Proposals logged and scored against outcomes |
| **M7 Multi-asset** | Interactive Brokers adapter, futures, FX, bonds; QuantLib curves and pricing | Bond and swap prices reconcile to a reference source |
