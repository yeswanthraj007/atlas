"""A thin wrapper over the Anthropic API: temperature 0, retries, token counts."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "claude-sonnet-5-5"
# Approximate dollars per million tokens (input, output), for the spend cap only. Override with
# ATLAS_LLM_PRICE_IN / ATLAS_LLM_PRICE_OUT if the real prices differ.
PRICE_IN = float(os.getenv("ATLAS_LLM_PRICE_IN", "3.0"))
PRICE_OUT = float(os.getenv("ATLAS_LLM_PRICE_OUT", "15.0"))


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int

    @property
    def cost(self) -> float:
        return (self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT) / 1e6


class LLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        max_tokens: int = 400,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._sleep = sleep
        if client is None:
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (add it to .env yourself)")
            import anthropic

            client = anthropic.Anthropic(api_key=key)
        self._client = client

    def complete(self, system: str, user: str) -> Completion:
        for attempt in range(4):
            try:
                r = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return Completion(text, int(r.usage.input_tokens), int(r.usage.output_tokens))
            except Exception:  # rate limit or transient error
                if attempt == 3:
                    raise
                self._sleep(5 * 2**attempt)
        raise RuntimeError("unreachable")
