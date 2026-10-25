import sys
import types

import numpy as np
import pandas as pd
import pytest

from atlas.lab.evaluate import CandidateSpec, Periods, evaluate_candidate
from atlas.lab.trials import Trial, TrialLog
from atlas.recommend.engine import Registry
from atlas.refdata.instruments import SecurityMaster
from atlas.research import ResearchData

RNG = np.random.default_rng(4)
PERIODS = Periods(dev_start="2017-01-01", dev_end="2022-12-31", holdout_start="2023-01-01")


def research_data(n: int = 120, strength: float = 0.004) -> ResearchData:
    """Synthetic market where a persistent characteristic `q` predicts returns."""
    idx = pd.bdate_range("2016-01-01", "2024-12-31", tz="UTC")
    cols = [f"EQ:S{i}" for i in range(n)]
    q = pd.DataFrame(RNG.normal(size=(len(idx), n)), index=idx, columns=cols)
    q = q.rolling(60, min_periods=1).mean()  # slow-moving, like a real characteristic
    rets = strength * q.shift(1).fillna(0) + RNG.normal(0, 0.015, (len(idx), n))
    closes = 100 * np.exp(rets.cumsum())
    members = pd.DataFrame(True, index=idx, columns=cols)
    return ResearchData(
        {}, SecurityMaster(), closes, closes, members, closes.mean(axis=1), {}, {"q": q}
    )


def install(
    name: str,
    compute,
    hypothesis: str = "stocks with high q beat low q next month",
    rationale: str = "investors underreact to slow-moving information in q",
) -> None:
    mod = types.ModuleType(f"atlas.candidates.{name}")
    mod.SPEC = CandidateSpec(name, hypothesis, rationale, 21, "test")
    mod.compute = compute
    sys.modules[f"atlas.candidates.{name}"] = mod


def honest(data: ResearchData) -> pd.DataFrame:
    return data.factors["q"]


def noise(data: ResearchData) -> pd.DataFrame:
    rng = np.random.default_rng(len(data.closes.columns))
    full = pd.DataFrame(rng.normal(size=(5000, data.closes.shape[1])), columns=data.closes.columns)
    return full.iloc[: len(data.closes)].set_axis(data.closes.index)


def cheater(data: ResearchData) -> pd.DataFrame:
    return data.closes.shift(-5) / data.closes - 1.0  # peeks 5 days into the future


@pytest.fixture
def lab(tmp_path):
    return TrialLog(tmp_path / "trials.jsonl"), tmp_path / "registry.json"


def test_real_signal_is_promoted_and_registered(lab) -> None:
    log, reg_path = lab
    install("cand_honest", honest)
    t = evaluate_candidate(
        "cand_honest", research_data(), log, reg_path, PERIODS, progress=lambda _: None
    )
    assert t.status == "promoted" and t.holdout_opened
    assert t.metrics["dev_ic_t"] > 3
    assert Registry.load(reg_path).validated("direction") == ["cand_honest"]


def test_noise_is_rejected_without_opening_the_holdout(lab) -> None:
    log, reg_path = lab
    install("cand_noise", noise)
    t = evaluate_candidate(
        "cand_noise", research_data(), log, reg_path, PERIODS, progress=lambda _: None
    )
    assert t.status == "rejected_dev" and not t.holdout_opened
    assert log.count() == 1 and log.holdout_views() == 0


def test_lookahead_is_caught(lab) -> None:
    log, reg_path = lab
    install("cand_cheat", cheater)
    t = evaluate_candidate(
        "cand_cheat", research_data(), log, reg_path, PERIODS, progress=lambda _: None
    )
    assert t.status == "rejected_lookahead"
    assert "change when later data is removed" in t.metrics["lookahead_check"]


def test_rationale_is_required_and_log_is_append_only(lab) -> None:
    log, reg_path = lab
    install("cand_lazy", honest, rationale="it works")
    with pytest.raises(ValueError, match="rationale"):
        evaluate_candidate("cand_lazy", research_data(), log, reg_path, PERIODS)
    log.append(Trial(1, "x", "test", "h" * 20, "r" * 20, "sha", "abc", "rejected_dev"))
    with pytest.raises(ValueError, match="append-only"):
        log.append(Trial(1, "y", "test", "h" * 20, "r" * 20, "sha", "abc", "promoted"))
    assert log.next_id() == 2
