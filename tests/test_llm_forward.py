# ruff: noqa: E501
import json

import pandas as pd

from atlas.llm_forward.client import LLMClient
from atlas.llm_forward.prompt import PROMPT_SHA, build_messages, parse_response
from atlas.llm_forward.session import is_eligible, reaction_session_start
from atlas.llm_forward.store import PredictionStore

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


def test_parser_accepts_only_exact_valid_replies():
    good = '{"p_reaction_up": 0.55, "p_drift_up": 0.5, "p_big_move": 0.7, "rationale": "x"}'
    assert parse_response(good)["p_big_move"] == 0.7
    assert parse_response("Sure!\n" + good + "\nDone")["p_reaction_up"] == 0.55  # fenced is fine
    assert parse_response('{"p_reaction_up": 1.2, "p_drift_up": 0.5, "p_big_move": 0.7}') is None
    assert parse_response('{"p_reaction_up": 0.5, "p_drift_up": 0.5}') is None
    assert parse_response('{"p_reaction_up": "0.5", "p_drift_up": 0.5, "p_big_move": 0.5}') is None
    assert parse_response('{"p_reaction_up": true, "p_drift_up": 0.5, "p_big_move": 0.5}') is None
    assert parse_response("no json here") is None
    assert len(PROMPT_SHA) == 64


def test_prompt_masks_nothing_itself_and_truncates_the_text():
    system, user = build_messages("x" * 10_000)
    assert len(user) < 6100 and "JSON" in system and "[COMPANY]" in system


def test_session_guard_after_close_before_open_and_during_hours():
    # filed 16:05 ET Tuesday: reaction at Wednesday's open (09:30 ET = 14:30 UTC in winter)
    after = ts("2026-11-10 21:05")
    assert reaction_session_start(after) == ts("2026-11-11 14:30")
    # filed 07:00 ET Wednesday: the same day's open
    assert reaction_session_start(ts("2026-11-11 12:00")) == ts("2026-11-11 14:30")
    # filed during market hours: never scored
    assert reaction_session_start(ts("2026-11-11 17:00")) is None
    # filed Friday after the close: Monday's open
    assert reaction_session_start(ts("2026-11-13 22:00")) == ts("2026-11-16 14:30")
    ok, _ = is_eligible(after, ts("2026-11-10 21:30"))
    assert ok
    ok, why = is_eligible(after, ts("2026-11-11 15:00"))  # after Wednesday's open
    assert not ok and "after" in why


def test_store_is_append_only_and_digests_change_if_anything_is_altered(tmp_path):
    s = PredictionStore(tmp_path / "p.jsonl")
    s.append(
        {"accession": "a1", "model": "m", "predicted_at": "2026-11-10T21:30:00+00:00", "p": 0.6}
    )
    s.append(
        {"accession": "a2", "model": "m", "predicted_at": "2026-11-10T22:00:00+00:00", "p": 0.4}
    )
    s.append(
        {"accession": "a3", "model": "m", "predicted_at": "2026-11-11T13:00:00+00:00", "p": 0.5}
    )
    assert s.seen("a1", "m") and not s.seen("a1", "other") and not s.seen("zz", "m")
    d = s.digest("2026-11-10")
    assert len(d) == 64 and s.digest("2026-11-12") == ""
    path = s.write_digest("2026-11-10", tmp_path / "hashes")
    assert "2 predictions" in path.read_text() and d in path.read_text()
    # tampering with an earlier line changes that day's fingerprint
    lines = s.lines()
    obj = json.loads(lines[0])
    obj["p"] = 0.99
    lines[0] = json.dumps(obj, sort_keys=True)
    (tmp_path / "p.jsonl").write_text("\n".join(lines) + "\n")
    assert s.digest("2026-11-10") != d
    assert len(s.records()) == 3


class _Msg:
    def __init__(self, text, i=100, o=50):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.usage = type("U", (), {"input_tokens": i, "output_tokens": o})()


class _Api:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        assert kw["temperature"] == 0
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Msg(r)


def test_client_retries_transient_errors_and_reports_cost():
    api = _Api([RuntimeError("rate limit"), RuntimeError("503"), "ok-reply"])
    c = LLMClient("m", client=api, sleep=lambda _: None)
    out = c.complete("sys", "user")
    assert out.text == "ok-reply" and api.calls == 3
    assert out.cost == (100 * 3.0 + 50 * 15.0) / 1e6


def test_outcomes_use_only_prices_after_the_release_and_wait_for_the_horizon():
    import numpy as np

    from atlas.llm_forward.score import compute_outcomes

    idx = pd.bdate_range("2026-01-01", periods=60, tz="UTC") + pd.Timedelta(hours=5)
    c = pd.DataFrame({"EQ:A": np.arange(100.0, 160.0)}, index=idx)
    spy = pd.Series(np.full(60, 100.0), index=idx)
    # filed 21:00 UTC (after the close) on the day of index 10 -> reaction session is index 11
    acc = pd.Series(
        [
            idx[10].normalize() + pd.Timedelta(hours=21, minutes=5),
            idx[50].normalize() + pd.Timedelta(hours=21, minutes=5),
        ]
    )
    out = compute_outcomes(acc, pd.Series(["EQ:A", "EQ:A"]), c, spy)
    assert list(out["d0"]) == [11, 51]
    assert np.isclose(out.loc[0, "react"], c.iloc[11, 0] / c.iloc[10, 0] - 1)
    assert np.isclose(out.loc[0, "drift"], c.iloc[31, 0] / c.iloc[11, 0] - 1)
    assert np.isnan(out.loc[1, "drift"])  # 20 more sessions do not exist yet
    c2 = c.copy()
    c2.iloc[:10] *= 9.0  # changing prices before the release cannot change the outcome
    out2 = compute_outcomes(acc, pd.Series(["EQ:A", "EQ:A"]), c2, spy)
    assert np.isclose(out2.loc[0, "drift"], out.loc[0, "drift"])


def test_scoring_prefers_an_informative_model_and_returns_nothing_when_unscoreable():
    import numpy as np

    from atlas.llm_forward.score import cluster_t, score_stream

    rng = np.random.default_rng(0)
    n = 600
    y3 = rng.integers(0, 2, n)
    react = np.where(y3 == 1, 0.08, 0.01) * rng.choice([-1, 1], n)
    df = pd.DataFrame({
        "p_reaction_up": rng.uniform(0.4, 0.6, n),
        "p_big_move": np.clip(0.5 + 0.3 * (y3 - 0.5) + rng.normal(0, 0.05, n), 0.02, 0.98),  # informative
        "tfidf_t1": rng.uniform(0.4, 0.6, n), "tfidf_t3": np.full(n, 0.5), "tfidf_t2": rng.normal(size=n),
        "p_drift_up": rng.uniform(0.4, 0.6, n), "react": react, "drift": rng.normal(0, 0.05, n),
        "quarter": 20261, "month": rng.choice(["2026-11", "2026-12"], n),
        "day": rng.integers(0, 40, n), "cost": 0.001,
    })  # fmt: skip
    s = score_stream(df, {20261: 0.04}, {"t1": 0.5, "t3": 0.5})
    assert s["t3"]["llm"]["log_loss"] < s["t3"]["tfidf"]["log_loss"]
    assert s["t3"]["paired_t_logloss_llm_minus_tfidf"] < -3  # negative: the model is better
    assert s["t3"]["llm"]["auc"] > 0.8 and abs(s["t1"]["llm"]["auc"] - 0.5) < 0.1
    assert score_stream(df.assign(react=np.nan), {20261: 0.04}, {"t1": 0.5, "t3": 0.5}) == {}
    assert np.isnan(cluster_t(np.ones(5), np.array([0, 0, 0, 0, 0])))  # one cluster: undefined
