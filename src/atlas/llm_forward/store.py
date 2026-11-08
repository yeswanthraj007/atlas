"""Append-only prediction log with daily fingerprints.

Predictions go to a JSONL file that is only ever appended to. The SHA-256 of each day's lines is
written to research/forward/hashes/<date>.txt; committing and pushing that file to git gives an
external timestamp: changing or back-dating a prediction later would change the hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path("data/llm_forward")
HASH_DIR = Path("research/forward/hashes")


class PredictionStore:
    def __init__(self, path: str | Path = ROOT / "predictions.jsonl") -> None:
        self.path = Path(path)

    def append(self, record: dict) -> str:
        """Add one prediction; returns its raw line. The log date is the UTC day of predicted_at."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return line

    def lines(self) -> list[str]:
        return self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []

    def records(self) -> pd.DataFrame:
        rows = [json.loads(x) for x in self.lines()]
        return pd.DataFrame(rows)

    def seen(self, accession: str, model: str) -> bool:
        return any(
            r["accession"] == accession and r["model"] == model
            for r in map(json.loads, self.lines())
        )

    def digest(self, day: str) -> str:
        """SHA-256 over the raw lines whose predicted_at falls on `day` (UTC, YYYY-MM-DD)."""
        mine = [x for x in self.lines() if json.loads(x)["predicted_at"][:10] == day]
        return hashlib.sha256("\n".join(mine).encode()).hexdigest() if mine else ""

    def write_digest(self, day: str, directory: str | Path = HASH_DIR) -> Path | None:
        h = self.digest(day)
        if not h:
            return None
        n = sum(1 for x in self.lines() if json.loads(x)["predicted_at"][:10] == day)
        p = Path(directory) / f"{day}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{day} {n} predictions sha256 {h}\n")
        return p
