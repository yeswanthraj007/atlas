"""The fixed prompt and the strict parser for the forward test. Changing anything here changes
PROMPT_SHA, which is stored with every prediction: a changed prompt is a different experiment."""

from __future__ import annotations

import hashlib
import json
import re

PROMPT_V1 = "v1"
TEXT_CHARS = 6000

SYSTEM = (
    "You will read the text of a US company's quarterly earnings press release. The company's "
    "name has been replaced by [COMPANY]. Estimate probabilities about the stock's behaviour "
    "after the release.\n"
    "Use only the text given and general financial knowledge. Do not use any knowledge of how "
    "this company's stock later behaved, or of events after the release date; if you think you "
    "recognise the company, ignore that. What moves a stock is results compared with what the "
    "market expected, and you cannot see expectations, so stay close to 0.5 unless the text gives "
    "a clear reason.\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    '{"p_reaction_up": <number>, "p_drift_up": <number>, "p_big_move": <number>, '
    '"rationale": "<at most 60 words>"}\n'
    "p_reaction_up: probability that the stock's market-adjusted return on the first trading "
    "session after the release is positive.\n"
    "p_drift_up: probability that its market-adjusted return from the close of that session to "
    "20 trading days later is positive.\n"
    "p_big_move: probability that the size of the first session's market-adjusted move is larger "
    "than the median for companies reporting this quarter.\n"
    "All probabilities are numbers between 0 and 1."
)
USER_TEMPLATE = "Earnings press release:\n\n{text}"
PROMPT_SHA = hashlib.sha256((PROMPT_V1 + SYSTEM + USER_TEMPLATE).encode()).hexdigest()

_KEYS = ("p_reaction_up", "p_drift_up", "p_big_move")


def build_messages(masked_text: str) -> tuple[str, str]:
    return SYSTEM, USER_TEMPLATE.format(text=masked_text[:TEXT_CHARS])


def parse_response(raw: str) -> dict | None:
    """The three probabilities (floats in [0, 1]) and a rationale, or None when the reply is not
    valid. Never repairs: a reply with a missing key or an out-of-range value is rejected."""
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out: dict = {}
    for k in _KEYS:
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, int | float) or not 0.0 <= float(v) <= 1.0:
            return None
        out[k] = float(v)
    out["rationale"] = str(d.get("rationale", ""))[:600]
    return out
