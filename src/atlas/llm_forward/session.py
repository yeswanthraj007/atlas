"""Was a prediction made before the market could react? A conservative check."""

from __future__ import annotations

import pandas as pd

NY = "America/New_York"


def reaction_session_start(accepted: pd.Timestamp) -> pd.Timestamp | None:
    """The 09:30 ET open of the first regular session that can react to a filing accepted at
    `accepted` (UTC). None when the filing arrived DURING market hours (the reaction is already
    under way, so the release is never scored). Uses weekdays only (not holidays): the true
    session can only be later than this, so the check never calls a late prediction early."""
    t = accepted.tz_convert(NY)
    open_, close = (
        t.normalize() + pd.Timedelta(hours=9, minutes=30),
        t.normalize() + pd.Timedelta(hours=16),
    )
    if t.weekday() < 5 and t < open_:
        start = open_
    elif t.weekday() < 5 and t < close:
        return None
    else:  # after the close, or on a weekend
        day = t.normalize() + pd.Timedelta(days=1)
        while day.weekday() >= 5:
            day += pd.Timedelta(days=1)
        start = day + pd.Timedelta(hours=9, minutes=30)
    return start.tz_convert("UTC")


def is_eligible(accepted: pd.Timestamp, predicted_at: pd.Timestamp) -> tuple[bool, str]:
    start = reaction_session_start(accepted)
    if start is None:
        return False, "filed during market hours"
    if predicted_at >= start:
        return False, "prediction made after the reaction session began"
    return True, "ok"
