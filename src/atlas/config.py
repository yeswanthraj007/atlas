"""Runtime settings, read from the environment (and an optional, git-ignored .env file).

Secrets are never passed as arguments, logged or written to disk by this code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class MissingCredentialsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    alpaca_key: str | None
    alpaca_secret: str | None
    alpaca_paper: bool
    alpaca_feed: str  # "iex" (free plan) or "sip" (paid, consolidated tape)
    data_dir: Path

    def require_alpaca(self) -> tuple[str, str]:
        if not self.alpaca_key or not self.alpaca_secret:
            raise MissingCredentialsError(
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment or in .env "
                "(see .env.example). Use paper-trading keys."
            )
        return self.alpaca_key, self.alpaca_secret


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    if env_file is not None and Path(env_file).exists():
        load_dotenv(env_file, override=False)
    return Settings(
        alpaca_key=os.getenv("ALPACA_API_KEY"),
        alpaca_secret=os.getenv("ALPACA_SECRET_KEY"),
        alpaca_paper=os.getenv("ALPACA_PAPER", "true").lower() != "false",
        alpaca_feed=os.getenv("ALPACA_DATA_FEED", "iex"),
        data_dir=Path(os.getenv("ATLAS_DATA_DIR", "data")),
    )
