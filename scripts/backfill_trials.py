"""One-off: record every experiment run before the trial log existed (2026-10-24/25).

Deflated Sharpe ratios must count ALL attempts, so earlier experiments are logged as
trials with what was measured at the time. Run once; the log refuses duplicate ids.

    uv run python scripts/backfill_trials.py
"""

from __future__ import annotations

from atlas.lab.trials import Trial, TrialLog

NOTE = "backfilled: run before the trial log existed; see git history and data/reports"
EARLIER = [
    # (candidate, status, metrics, holdout_opened, hypothesis)
    *[
        (
            f"momentum_wf_lookback{lb}_top{n}",
            "rejected_holdout",
            {"oos_ic": -0.021, "oos_ic_t": -0.95, "alpha_t_vs_spy": None},
            True,
            "12-1 style momentum picks outperform the S&P 500",
        )
        for lb in (126, 252)
        for n in (10, 20, 40)
    ],
    *[
        (
            f"multifactor_{w}_top{n}_trend{tf}",
            "rejected_holdout",
            {"oos_ic": -0.009, "oos_ic_t": -0.44, "alpha_t_vs_spy": -0.23},
            True,
            "IC-weighted price-factor composite outperforms",
        )
        for w in ("ic", "equal")
        for n in (20, 40)
        for tf in (False, True)
    ],
    *[
        (
            f"ml_return_rank_{m}_{f}",
            "rejected_holdout",
            {"holdout_ic": ic, "holdout_ic_t": t},
            True,
            "gradient boosting / ridge on factor ranks predicts monthly return ranks",
        )
        for (m, f, ic, t) in [
            ("ridge", "all", -0.013, -0.58),
            ("gbm_shallow", "all", -0.013, -0.58),
            ("gbm", "all", -0.007, -0.26),
            ("ridge", "extended", 0.010, 0.37),
            ("gbm_shallow", "extended", 0.010, 0.37),
            ("gbm", "extended", 0.010, 0.37),
        ]
    ],
    *[
        (
            f"single_factor_{name}",
            "rejected_dev",
            {"oos_ic": ic, "oos_ic_t": t},
            True,
            f"{name} predicts next-month relative returns",
        )
        for (name, ic, t) in [
            ("mom_12_1", -0.013, -0.5),
            ("vol_adj_mom", -0.009, -0.4),
            ("reversal_1m", -0.012, -0.6),
            ("low_vol", -0.041, -1.3),
            ("high_52w", -0.015, -0.6),
            ("earnings_yield", 0.012, 0.7),
            ("book_to_price", 0.003, 0.1),
            ("sales_to_price", 0.022, 1.1),
            ("cf_yield", 0.012, 0.5),
            ("gross_profitability", 0.008, 0.5),
            ("roe", 0.008, 0.4),
            ("low_accruals", 0.008, 0.8),
            ("low_asset_growth", -0.006, -0.5),
            ("low_leverage", -0.026, -1.7),
            ("revenue_growth", 0.008, 0.5),
        ]
    ],
    *[
        (
            f"lazy_prices_{sig}_{h}d",
            "rejected_dev",
            {"dev_ic": ic, "dev_ic_t": t},
            True,
            "firms whose filings change little outperform big changers",
        )
        for (sig, h, ic, t) in [
            ("cosine", 21, 0.011, 1.28),
            ("jaccard", 21, 0.012, 1.48),
            ("risk_factors", 21, 0.011, 1.37),
            ("length", 21, 0.002, 0.33),
            ("cosine", 63, 0.017, 1.12),
            ("jaccard", 63, 0.027, 1.94),
            ("risk_factors", 63, 0.021, 1.36),
            ("length", 63, -0.004, -0.34),
        ]
    ],
    (
        "vol_managed_spy",
        "rejected_holdout",
        {"sharpe": 0.83, "spy_sharpe": 0.82, "alpha_t": 0.43},
        True,
        "volatility-targeted SPY improves risk-adjusted returns",
    ),
    (
        "risk_balanced_stocks",
        "rejected_holdout",
        {"sharpe": 0.69, "alpha_t": -0.18},
        True,
        "inverse forecast-vol stock portfolio improves risk-adjusted returns",
    ),
]


def main() -> None:
    log = TrialLog()
    if log.count():
        raise SystemExit(f"log already has {log.count()} trials; backfill runs only once")
    for i, (name, status, metrics, opened, hyp) in enumerate(EARLIER, 1):
        log.append(
            Trial(
                trial_id=i,
                candidate=name,
                author="claude",
                hypothesis=hyp,
                rationale="see the corresponding validation script and report",
                code_sha256="n/a",
                git_commit="pre-log",
                status=status,
                metrics=metrics,
                holdout_opened=opened,
                created_at="2026-10-25T00:00:00+00:00",
                notes=NOTE,
            )
        )
    print(f"backfilled {log.count()} earlier trials")


if __name__ == "__main__":
    main()
