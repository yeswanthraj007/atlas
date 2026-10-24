# Atlas

An institutional-grade portfolio, risk and trading platform, modelled on the architecture
of BlackRock Aladdin and built by one person.

**Principle:** one security master, one append-only ledger, one position book. Every
module (risk, optimization, the order system, strategies, the UI) reads the same data,
and every order goes through the same pre-trade risk checks.

## Modules

| Package | What it does |
| --- | --- |
| `atlas.refdata` | Security master: equities, ETFs, options (OCC symbols), futures, FX, bonds |
| `atlas.ledger` | Append-only, bitemporal transaction ledger; positions are rebuilt from it at any date |
| `atlas.risk` | Historical, parametric and Monte Carlo (Student-t) VaR/ES; component VaR; EWMA and Ledoit-Wolf covariance; factor models; stress tests; Kupiec/Christoffersen backtests |
| `atlas.pricing` | Black-Scholes-Merton prices, Greeks and implied volatility |
| `atlas.optimizer` | Mean-variance with caps, vol targets, turnover and cost limits; risk parity |
| `atlas.oms` | Order state machine, pre-trade risk rules, kill switch, audit log |
| `atlas.execution` | Broker interface and a pessimistic simulated broker |
| `atlas.strategy` | Strategy interface, target-weight rebalancer, reference momentum strategy |
| `atlas.backtest` | Event-driven backtester that runs through the production order system |

## Getting started

```bash
uv sync
uv run pytest
```

## Safety

Strategies never talk to a broker directly. The order system rejects any order that
breaks a risk rule, and a broken rule rejects rather than lets an order through. Live
trading stays off until a strategy has passed a backtest and a paper-trading period.
