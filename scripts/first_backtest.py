"""First real-data backtest: S&P 500 momentum vs SPY buy-and-hold, 2016 to today.

Caveat printed with the results: the universe is *today's* S&P 500, so companies that
were removed or went bankrupt since 2016 are missing (survivorship bias). Results are
therefore optimistic and are a test of the machinery, not evidence of an edge.

    uv run python scripts/first_backtest.py
"""

from __future__ import annotations

import time
from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.backtest.engine import BacktestConfig, performance_metrics, run_backtest
from atlas.execution.broker import CostModel
from atlas.marketdata.loader import load_for_backtest
from atlas.marketdata.store import BarStore
from atlas.oms.risk_checks import MaxGrossExposure, MaxPositionWeight, NoShortSelling
from atlas.refdata.instruments import SecurityMaster
from atlas.risk import backtesting as vbt
from atlas.risk.var import filtered_historical_var, var_es_from_pnl
from atlas.strategy.base import StrategyContext
from atlas.strategy.momentum import CrossSectionalMomentum

COSTS = CostModel(per_share=D(0), min_per_order=D(0), bps_of_notional=D(1), slippage_bps=D(5))


def config(max_position: str = "0.10") -> BacktestConfig:
    return BacktestConfig(
        initial_cash=D(1_000_000),
        costs=COSTS,
        max_participation=0.01,  # never more than 1% of a day's volume
        rules=lambda: [
            MaxPositionWeight(D(max_position)),
            MaxGrossExposure(D("1.00")),
            NoShortSelling(),
        ],
    )


class EqualWeight:
    """Every stock with a price, equal weight, rebalanced monthly. Shares the universe's
    survivorship bias, so momentum minus this isolates what the signal itself adds."""

    id = "equal_weight"

    def on_bar(self, ctx: StrategyContext) -> dict[str, float] | None:
        if ctx.bar_index % 21 != 0 or ctx.bar_index < 252:
            return None
        names = list(ctx.history.iloc[-1].dropna().index)
        return {n: 0.98 / len(names) for n in names}


class BuyAndHold:
    id = "buy_and_hold"

    def __init__(self, inst: str) -> None:
        self.inst = inst

    def on_bar(self, ctx: StrategyContext) -> dict[str, float] | None:
        return {self.inst: 0.98} if ctx.bar_index == 0 else None


def rolling_var_backtest(equity: pd.Series, window: int = 500, conf: float = 0.99) -> pd.DataFrame:
    """Each day, forecast tomorrow's 99% VaR from the previous `window` days only, then
    count how often the realised loss exceeded it. Plain historical vs filtered (FHS)."""
    rets = equity.pct_change().dropna().to_numpy()
    nav = equity.to_numpy()[1:]
    plain, fhs, realised = [], [], []
    for t in range(window, len(rets)):
        hist = rets[t - window : t]
        value = nav[t - 1]  # book value at the moment the forecast is made
        plain.append(var_es_from_pnl(hist * value, conf)[0])
        fhs.append(filtered_historical_var(hist, value, conf).var)
        realised.append(rets[t] * value)
    rows = {}
    for name, fc in (("historical", plain), ("filtered_hs", fhs)):
        hits = vbt.exceptions(np.array(realised), np.array(fc))
        rows[name] = {
            "days": len(hits),
            "exceptions": int(hits.sum()),
            "expected": round(len(hits) * (1 - conf), 1),
            "kupiec_p": round(vbt.kupiec_pof(hits, conf).p_value, 3),
            "independence_p": round(vbt.christoffersen_independence(hits).p_value, 3),
            "basel_last_250": vbt.basel_traffic_light(hits).value,
        }
    return pd.DataFrame(rows)


def main() -> None:
    store = BarStore("data")
    stocks = [s.strip() for s in Path("data/universes/sp500.txt").read_text().split()]

    sm = SecurityMaster()
    t0 = time.time()
    bars = load_for_backtest(store, stocks, sm)
    spy_sm = SecurityMaster()
    spy = load_for_backtest(store, ["SPY"], spy_sm, etfs={"SPY"})
    print(f"loaded {len(bars)} stocks in {time.time() - t0:.1f}s")

    t0 = time.time()
    strat = CrossSectionalMomentum(top_n=20, rebalance_every=21, max_weight=0.08, gross=0.98)
    res = run_backtest(strat, bars, sm, config())
    print(
        f"momentum backtest: {time.time() - t0:.1f}s, {len(res.ledger) - 1} trades, "
        f"{len(res.rejected)} orders rejected by risk rules"
    )
    ew = run_backtest(EqualWeight(), bars, sm, config())
    bench = run_backtest(BuyAndHold("ETF:SPY"), spy, spy_sm, config(max_position="1.00"))
    for name, r in (("equal_weight", ew), ("spy", bench)):
        if r.rejected:
            print(f"{name}: {len(r.rejected)} rejected, e.g. {r.rejected[0].reject_reasons[0]}")

    # Compare on the momentum strategy's live period (it needs a year of history first)
    first_trade = min(t.effective_at for t in res.ledger.transactions() if t.instrument_id)
    eq = res.equity.loc[first_trade:]
    curves = {
        "momentum_top20": eq,
        "equal_weight_same_stocks": ew.equity.loc[first_trade:],
        "spy_buy_hold": bench.equity.loc[first_trade:],
    }
    table = pd.DataFrame({k: performance_metrics(v) for k, v in curves.items()})
    fmt = {"sharpe": "{:.2f}", "sortino": "{:.2f}", "calmar": "{:.2f}"}
    shown = table.copy().astype(object)
    for metric in table.index:
        f = fmt.get(metric, "{:.1%}")
        shown.loc[metric] = [f.format(x) for x in table.loc[metric]]
    print(f"\nperiod: {eq.index[0].date()} to {eq.index[-1].date()}")
    print(shown.to_string())
    print("\nVaR model validation on the momentum book (99%, 1-day, 500-day window):")
    print(rolling_var_backtest(eq).to_string())
    print("\nCAVEAT: universe is today's S&P 500 (survivorship bias): results are optimistic.")

    out = Path("data/reports")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(out / "first_backtest_equity.csv")


if __name__ == "__main__":
    main()
