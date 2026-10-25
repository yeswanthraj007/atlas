"""Volatility-managed portfolios, tested out of sample.

    uv run python scripts/holdout_volmanaged.py

Protocol 1 (frozen): volatility models trained on data known by 2024-12-31, then used
unchanged for 2025-2026.
Protocol 2 (yearly retrain): for each year Y from 2020, models are retrained on data
known by Dec 31 of Y-1 and used for year Y only. Every forecast is out of sample.
"""

from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.backtest.engine import BacktestConfig, performance_metrics, run_backtest
from atlas.execution.broker import CostModel
from atlas.marketdata.loader import load_for_backtest
from atlas.marketdata.store import BarStore
from atlas.ml.models import RidgeRanker
from atlas.ml.volatility import HAR_FEATURES, build_vol_dataset
from atlas.oms.risk_checks import MaxGrossExposure, MaxPositionWeight, NoShortSelling
from atlas.refdata.instruments import SecurityMaster
from atlas.research import load_research_data
from atlas.strategy.base import StrategyContext
from atlas.strategy.vol_managed import EqualWeightStocks, RiskBalancedStocks, VolTargetIndex
from atlas.validation.stats import alpha_beta

H = 21
COSTS = CostModel(per_share=D(0), min_per_order=D(0), bps_of_notional=D(1), slippage_bps=D(5))
OUT = Path("data/reports/vol_managed")


def config(max_pos: str) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=D(1_000_000),
        costs=COSTS,
        max_participation=0.01,
        rules=lambda: [
            MaxPositionWeight(D(max_pos)),
            MaxGrossExposure(D("1.00")),
            NoShortSelling(),
        ],
    )


class BuyHold:
    id = "buy_hold"

    def on_bar(self, ctx: StrategyContext) -> dict[str, float] | None:
        return {"ETF:SPY": 0.98} if ctx.bar_index == 0 else None


def forecast_frames(
    closes: pd.DataFrame,
    cutoffs: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    every_train: int = 5,
) -> pd.DataFrame:
    """HAR forecasts (annualised vol) per instrument. `cutoffs`: (train_end, use_from,
    use_to). Each model sees only labels known by its train_end."""
    idx = closes.index
    out = pd.DataFrame(np.nan, index=idx, columns=closes.columns)
    for train_end, start, end in cutoffs:
        grid = idx[(idx >= idx[0] + pd.Timedelta(days=120)) & (idx <= train_end)][::every_train]
        use = idx[(idx >= start) & (idx <= end)]
        data = build_vol_dataset(closes, grid.union(use), H)
        le = pd.to_datetime(data["label_end"])
        train = data[(le <= train_end) & data["target"].notna()]
        model = RidgeRanker(alpha=1e-6).fit(train[HAR_FEATURES], train["target"])
        test = data[data.index.get_level_values("date").isin(use)]
        pred = pd.Series(np.exp(model.predict(test[HAR_FEATURES])), index=test.index)
        out.update(pred.unstack("instrument"))
    return out


def evaluate(
    label: str, start: pd.Timestamp, end: pd.Timestamp, cutoffs, rd, spy_bars, spy_sm, spy_close
) -> pd.DataFrame:
    spy_closes = spy_close.to_frame("ETF:SPY")
    spy_fc = forecast_frames(spy_closes, cutoffs)["ETF:SPY"]
    stock_fc = forecast_frames(rd.closes, cutoffs)
    train_end = cutoffs[0][0]
    r = spy_close.pct_change().loc[:train_end]
    target_vol = float((r.rolling(21).std() * np.sqrt(252)).median())
    warm = pd.Timedelta(days=10)

    def window(bars):
        return {k: v.loc[start - warm : end] for k, v in bars.items() if len(v.loc[start:end])}

    runs = {
        "SPY buy & hold": run_backtest(
            BuyHold(), window(spy_bars), spy_sm, config("1.00"), start_trading=start
        ),
        "A: SPY vol-targeted": run_backtest(
            VolTargetIndex("ETF:SPY", spy_fc, target_vol),
            window(spy_bars),
            spy_sm,
            config("1.00"),
            start_trading=start,
        ),
        "Equal-weight S&P": run_backtest(
            EqualWeightStocks(),
            window(rd.bars),
            rd.secmaster,
            config("0.10"),
            membership=rd.membership,
            start_trading=start,
        ),
        "B: risk-balanced stocks": run_backtest(
            RiskBalancedStocks(stock_fc, spy_fc, target_vol),
            window(rd.bars),
            rd.secmaster,
            config("0.10"),
            membership=rd.membership,
            start_trading=start,
        ),
    }
    bench = runs["SPY buy & hold"].equity.loc[start:end].pct_change().dropna()
    rows = {}
    for name, res in runs.items():
        eq = res.equity.loc[start:end]
        m = performance_metrics(eq)
        ab = alpha_beta(
            eq.pct_change().dropna().to_numpy(),
            bench.reindex(eq.pct_change().dropna().index).fillna(0).to_numpy(),
        )
        rows[name] = {**m, "beta": ab["beta"], "alpha": ab["alpha_ann"], "alpha_t": ab["alpha_t"]}
    table = pd.DataFrame(rows).T[
        ["cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "beta", "alpha", "alpha_t"]
    ]
    print(f"\n=== {label}: {start.date()} .. {end.date()} | target vol {target_vol:.1%} ===")
    fmt = {
        "cagr": "{:.1%}",
        "ann_vol": "{:.1%}",
        "sharpe": "{:.2f}",
        "max_drawdown": "{:.1%}",
        "calmar": "{:.2f}",
        "beta": "{:.2f}",
        "alpha": "{:+.1%}",
        "alpha_t": "{:+.2f}",
    }
    print(table.apply(lambda c: c.map(fmt[c.name].format)).to_string())
    expo = spy_fc.loc[start:end].map(lambda f: min(1.0, target_vol / f))
    print(
        f"strategy A exposure: avg {expo.mean():.0%}, min {expo.min():.0%} "
        f"(on {expo.idxmin().date()}), at 100% on {float((expo >= 0.999).mean()):.0%} of days"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: v.equity.loc[start:end] for k, v in runs.items()}).to_csv(
        OUT / f"{label.split()[0].lower()}_equity.csv"
    )
    return table


def main() -> None:
    rd = load_research_data("price")
    store = BarStore("data")
    spy_sm = SecurityMaster()
    spy_bars = load_for_backtest(store, ["SPY"], spy_sm, etfs={"SPY"})
    spy_close = spy_bars["ETF:SPY"]["close"]
    last = rd.closes.index[-1]
    tz = "UTC"

    frozen = [(pd.Timestamp("2024-12-31", tz=tz), pd.Timestamp("2025-01-02", tz=tz), last)]
    evaluate("FROZEN test", frozen[0][1], last, frozen, rd, spy_bars, spy_sm, spy_close)

    yearly = []
    for y in range(2020, last.year + 1):
        yearly.append(
            (
                pd.Timestamp(f"{y - 1}-12-31", tz=tz),
                pd.Timestamp(f"{y}-01-01", tz=tz),
                min(pd.Timestamp(f"{y}-12-31", tz=tz), last),
            )
        )
    evaluate("YEARLY-RETRAIN test", yearly[0][1], last, yearly, rd, spy_bars, spy_sm, spy_close)
    print(
        "\nAlpha with |t| < 2 is not evidence of skill. Vol targeting aims for better "
        "risk-adjusted returns and smaller drawdowns, not higher raw returns."
    )


if __name__ == "__main__":
    main()
