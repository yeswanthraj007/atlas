"""Append-only trial log: every signal idea ever evaluated, including every failure.

Testing 50 ideas and keeping the best one looks like skill when it is luck. The deflated
Sharpe ratio corrects for that only if every attempt is counted, so this log is never
edited or truncated: a corrected idea is a NEW trial, and the old one stays.

Stored as JSON lines in research/trials.jsonl, which is committed to git, so the history
of attempts is itself version-controlled and auditable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_PATH = Path("research/trials.jsonl")


@dataclass
class Trial:
    trial_id: int
    candidate: str
    author: str
    hypothesis: str
    rationale: str
    code_sha256: str
    git_commit: str
    status: (
        str  # rejected_lookahead | rejected_dev | rejected_holdout | rejected_deflated | promoted
    )
    metrics: dict[str, Any] = field(default_factory=dict)
    holdout_opened: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    notes: str = ""


class TrialLog:
    def __init__(self, path: str | Path = LOG_PATH) -> None:
        self.path = Path(path)

    def all(self) -> list[Trial]:
        if not self.path.exists():
            return []
        rows = [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]
        return [Trial(**r) for r in rows]

    def next_id(self) -> int:
        trials = self.all()
        return max((t.trial_id for t in trials), default=0) + 1

    def append(self, trial: Trial) -> Trial:
        """The only write operation. There is deliberately no update or delete."""
        existing = {t.trial_id for t in self.all()}
        if trial.trial_id in existing:
            raise ValueError(f"trial {trial.trial_id} already logged; the log is append-only")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(trial), default=str) + "\n")
        return trial

    def count(self) -> int:
        return len(self.all())

    def dev_sharpes(self) -> list[float]:
        """Per-period Sharpe of every trial's development long-short returns."""
        return [
            float(t.metrics["dev_ls_sharpe_per_period"])
            for t in self.all()
            if t.metrics.get("dev_ls_sharpe_per_period") is not None
        ]

    def holdout_views(self) -> int:
        return sum(t.holdout_opened for t in self.all())


def log_external_trial(
    candidate: str,
    status: str,
    hypothesis: str,
    rationale: str,
    metrics: dict[str, Any],
    holdout_opened: bool,
    script: str,
    notes: str = "",
    log: TrialLog | None = None,
) -> Trial:
    """Record a trial run by a dedicated script (signals that train their own model)."""
    import hashlib
    import subprocess

    log = log or TrialLog()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    sha = hashlib.sha256(Path(script).read_bytes()).hexdigest() if Path(script).exists() else ""
    return log.append(
        Trial(
            trial_id=log.next_id(),
            candidate=candidate,
            author="claude",
            hypothesis=hypothesis,
            rationale=rationale,
            code_sha256=sha,
            git_commit=commit,
            status=status,
            metrics=metrics,
            holdout_opened=holdout_opened,
            notes=notes,
        )
    )
