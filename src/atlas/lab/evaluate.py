"""Evaluate a candidate signal through fixed gates, log the trial, maybe promote it.

A candidate is a module in atlas/candidates/ with:
    SPEC = CandidateSpec(name=..., hypothesis=..., rationale=..., horizon_days=21, author=...)
    def compute(data: ResearchData) -> pd.DataFrame   # dates x "EQ:<ticker>", higher = better

Gates, in order (the first failure ends the evaluation and is logged):
  1. Look-ahead: recompute on data truncated at earlier dates; values on each cut date
     must match the full-data values exactly.
  2. Development (dev period, default 2017..2024): IC t-stat >= 3 (Harvey, Liu & Zhu)
     and positive IC in at least 60% of calendar years.
  3. Frozen holdout (2025..): opened ONLY if gate 2 passed. IC must have the dev sign
     with t >= 2.
  4. Deflated Sharpe: the long-short spread's Sharpe must stay significant (P >= 0.95)
     after accounting for EVERY trial in the log.
Passing all four writes a "direction" entry to the signal registry.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import subprocess
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from atlas.lab.trials import Trial, TrialLog
from atlas.recommend.engine import Registry, SignalRecord
from atlas.research import ResearchData
from atlas.validation.stats import deflated_sharpe

DEV_T = 3.0
DEV_YEARS_POSITIVE = 0.6
HOLDOUT_T = 2.0
DSR_MIN = 0.95


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    hypothesis: str  # what should happen, stated before testing
    rationale: str  # WHY it should work (an economic reason, not "the data showed it")
    horizon_days: int = 21
    author: str = "claude"
    notes: str = ""  # disclose anything that weakens the test, e.g. a holdout seen before


@dataclass(frozen=True)
class Periods:
    dev_start: str = "2017-01-01"
    dev_end: str = "2024-12-31"
    holdout_start: str = "2025-01-01"


def truncate(rd: ResearchData, cut: pd.Timestamp) -> ResearchData:
    """The research data exactly as it existed at the close of `cut`."""
    return replace(
        rd,
        bars={k: v.loc[:cut] for k, v in rd.bars.items()},
        closes=rd.closes.loc[:cut],
        split_closes=rd.split_closes.loc[:cut],
        membership=rd.membership.loc[:cut],
        benchmark=rd.benchmark.loc[:cut],
        factors={k: v.loc[:cut] for k, v in rd.factors.items()},
    )


def lookahead_check(
    compute,
    rd: ResearchData,
    full: pd.DataFrame,
    n_cuts: int = 3,
) -> tuple[bool, str]:
    idx = full.dropna(how="all").index
    if len(idx) < 50:
        return False, "signal has too few dates to check"
    cuts = [idx[int(len(idx) * f)] for f in np.linspace(0.3, 0.9, n_cuts)]
    for cut in cuts:
        part = compute(truncate(rd, cut))
        a = full.loc[cut].reindex(full.columns)
        b = part.loc[cut].reindex(full.columns) if cut in part.index else a * np.nan
        same = np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=1e-9)
        if not same.all():
            bad = list(a.index[~same][:3])
            return False, f"values on {cut.date()} change when later data is removed: {bad}"
    return True, f"identical on {len(cuts)} cut dates"


@dataclass
class ICResult:
    ic: pd.Series  # Spearman IC per date
    spread: pd.Series  # top-decile minus bottom-decile return per date (gross)
    cost: pd.Series  # round-trip cost of both legs per date (Corwin-Schultz), if known
    by_segment: dict[str, pd.Series]  # IC per date within each named segment

    @property
    def net(self) -> pd.Series:
        return self.spread - self.cost.reindex(self.spread.index).fillna(0.0)


def _ic_series(
    signal: pd.DataFrame,
    closes: pd.DataFrame,
    membership: pd.DataFrame,
    dates: pd.DatetimeIndex,
    horizon: int,
    spread_cost: pd.DataFrame | None = None,
    segments: dict[str, set[str]] | None = None,
) -> ICResult:
    """Cross-sectional scoring on each date. Costs assume the whole decile portfolio is
    replaced each period (both legs pay a full round-trip spread): deliberately pessimistic."""
    fwd = closes.shift(-horizon) / closes - 1.0
    ics, spreads, costs = {}, {}, {}
    seg_ics: dict[str, dict] = {k: {} for k in (segments or {})}
    for d in dates:
        s = signal.loc[d] if d in signal.index else None
        if s is None:
            continue
        m = membership.loc[d] if d in membership.index else None
        both = pd.concat([s, fwd.loc[d]], axis=1, keys=["s", "r"]).dropna()
        if m is not None:
            both = both[both.index.isin(m[m].index)]
        if len(both) < 30 or both["s"].nunique() < 5:
            continue
        ics[d] = stats.spearmanr(both["s"], both["r"]).statistic
        q = pd.qcut(both["s"].rank(method="first"), 10, labels=False)
        top, bottom = both.index[q == 9], both.index[q == 0]
        spreads[d] = both["r"][top].mean() - both["r"][bottom].mean()
        if spread_cost is not None and d in spread_cost.index:
            c = spread_cost.loc[d]
            costs[d] = float(c.reindex(top).mean() + c.reindex(bottom).mean())
        for name, members in (segments or {}).items():
            sub = both[both.index.isin(members)]
            if len(sub) >= 30 and sub["s"].nunique() >= 5:
                seg_ics[name][d] = stats.spearmanr(sub["s"], sub["r"]).statistic
    return ICResult(
        pd.Series(ics, dtype=float),
        pd.Series(spreads, dtype=float),
        pd.Series(costs, dtype=float),
        {k: pd.Series(v, dtype=float) for k, v in seg_ics.items()},
    )


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() else 0.0


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def evaluate_candidate(
    name: str,
    rd: ResearchData,
    log: TrialLog | None = None,
    registry_path=None,
    periods: Periods | None = None,
    progress=print,
    segments: dict[str, set[str]] | None = None,
    universe: str = "sp500",
) -> Trial:
    log = log or TrialLog()
    periods = periods or Periods()
    module = importlib.import_module(f"atlas.candidates.{name}")
    spec: CandidateSpec = module.SPEC
    for fld in ("hypothesis", "rationale"):
        if len(getattr(spec, fld).strip()) < 20:
            raise ValueError(f"candidate {name}: write a real {fld} before testing it")
    try:
        code = inspect.getsource(module)
    except (OSError, TypeError):  # module built in memory (tests): hash the function
        code = inspect.getsource(module.compute)
    trial = Trial(
        trial_id=log.next_id(),
        candidate=spec.name,
        author=spec.author,
        hypothesis=spec.hypothesis,
        rationale=spec.rationale,
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        git_commit=_git_commit(),
        status="pending",
        notes=spec.notes,
    )
    h = spec.horizon_days
    progress(f"trial #{trial.trial_id}: {spec.name} (horizon {h}d, by {spec.author})")

    full = module.compute(rd).reindex(columns=rd.closes.columns)
    ok, msg = lookahead_check(module.compute, rd, full)
    trial.metrics["lookahead_check"] = msg
    progress(f"  gate 1 look-ahead: {'pass' if ok else 'FAIL'} ({msg})")
    if not ok:
        trial.status = "rejected_lookahead"
        return log.append(trial)

    idx = rd.closes.index
    dev = idx[
        (idx >= pd.Timestamp(periods.dev_start, tz="UTC"))
        & (idx <= pd.Timestamp(periods.dev_end, tz="UTC"))
    ][::h]
    dev = dev[idx.get_indexer(dev) + h < len(idx)]
    dres = _ic_series(full, rd.closes, rd.membership, dev, h, rd.spread, segments)
    ic, ls = dres.ic, dres.spread
    yearly = ic.groupby(ic.index.year).mean()
    dev_t = _t(ic)
    sign = 1.0 if ic.mean() >= 0 else -1.0
    years_ok = float((np.sign(yearly) == sign).mean()) if len(yearly) else 0.0
    ls_sharpe = float(ls.mean() / ls.std(ddof=1)) if len(ls) > 2 and ls.std() else 0.0
    trial.metrics.update(
        {
            "dev_ic": float(ic.mean()),
            "dev_ic_t": dev_t,
            "dev_dates": len(ic),
            "dev_years_same_sign": years_ok,
            "dev_spread": float(ls.mean()),
            "dev_ls_sharpe_per_period": ls_sharpe * sign,
            "dev_yearly_ic": {str(k): round(float(v), 4) for k, v in yearly.items()},
            "universe": universe,
            "dev_cost": float(dres.cost.mean()) if len(dres.cost) else None,
            # trade in the signal's direction; costs are always paid, whatever the sign
            "dev_spread_net": float(
                (dres.spread * sign - dres.cost.reindex(dres.spread.index).fillna(0.0)).mean()
            )
            if len(dres.cost)
            else None,
            "dev_spread_at_cost": {
                f"{bps}bps": float((dres.spread * sign).mean() - 2 * bps / 1e4)
                for bps in (0, 10, 30)
            },
            "dev_segments": {
                k: {"ic": float(v.mean()), "t": _t(v), "dates": len(v)}
                for k, v in dres.by_segment.items()
                if len(v)
            },
        }
    )
    for k, v in trial.metrics["dev_segments"].items():
        progress(f"    segment {k}: IC {v['ic']:+.4f}, t {v['t']:+.2f} over {v['dates']} dates")
    if trial.metrics["dev_cost"] is not None:
        progress(
            f"    decile spread {float(ls.mean()) * sign:+.2%} gross, "
            f"{trial.metrics['dev_spread_net']:+.2%} after an estimated "
            f"{trial.metrics['dev_cost']:.2%} round-trip cost per period"
        )
    progress(
        f"  gate 2 development: IC {ic.mean():+.4f}, t {dev_t:+.2f}, same sign in "
        f"{years_ok:.0%} of years"
    )
    if abs(dev_t) < DEV_T or years_ok < DEV_YEARS_POSITIVE:
        trial.status = "rejected_dev"
        progress("  -> rejected at development; the frozen holdout stays unopened")
        return log.append(trial)

    trial.holdout_opened = True
    hold = idx[idx >= pd.Timestamp(periods.holdout_start, tz="UTC")][::h]
    hold = hold[idx.get_indexer(hold) + h < len(idx)]
    hres = _ic_series(full, rd.closes, rd.membership, hold, h, rd.spread, segments)
    hic, hls = hres.ic, hres.spread
    hold_t = _t(hic) * sign
    trial.metrics.update(
        {
            "holdout_ic": float(hic.mean()),
            "holdout_ic_t_signed": hold_t,
            "holdout_dates": len(hic),
            "holdout_spread": float(hls.mean()),
            "holdout_spread_net": float(
                (hres.spread * sign - hres.cost.reindex(hres.spread.index).fillna(0.0)).mean()
            )
            if len(hres.cost)
            else None,
            "holdout_segments": {
                k: {"ic": float(v.mean()), "t": _t(v), "dates": len(v)}
                for k, v in hres.by_segment.items()
                if len(v)
            },
        }
    )
    progress(f"  gate 3 frozen holdout: IC {hic.mean():+.4f}, t (dev sign) {hold_t:+.2f}")
    net = trial.metrics["holdout_spread_net"]
    if hold_t < HOLDOUT_T or (net is not None and net <= 0):
        if net is not None:
            progress(f"    holdout decile spread after costs: {net:+.2%} per period")
        trial.status = "rejected_holdout"
        return log.append(trial)

    all_ls = pd.concat([ls, hls]).to_numpy() * sign
    n_trials = log.count() + 1
    dsr = deflated_sharpe(all_ls, n_trials, np.array([*log.dev_sharpes(), ls_sharpe]))
    trial.metrics.update({"deflated_sharpe_prob": dsr, "trials_counted": n_trials})
    progress(f"  gate 4 deflated Sharpe: P(real) {dsr:.3f} after {n_trials} trials")
    if dsr < DSR_MIN:
        trial.status = "rejected_deflated"
        return log.append(trial)

    trial.status = "promoted"
    reg = Registry.load(registry_path) if registry_path else Registry.load()
    reg.record(
        SignalRecord(
            name=spec.name,
            kind="direction",
            description=spec.hypothesis,
            holdout_metric=f"ic_{h}d",
            holdout_value=float(hic.mean()),
            holdout_t=hold_t,
            tested_on=f"holdout from {periods.holdout_start}",
            notes=f"trial #{trial.trial_id}; deflated Sharpe P={dsr:.3f} over {n_trials} trials",
        )
    )
    if registry_path:
        reg.save(registry_path)
    else:
        reg.save()
    progress("  -> PROMOTED: recorded in the signal registry")
    return log.append(trial)
