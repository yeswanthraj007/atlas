"""Network safety for long jobs."""

from __future__ import annotations

import requests

_INSTALLED = False


def install_default_timeout(seconds: float = 90.0) -> None:
    """Give every `requests` call a timeout unless it sets its own. The Alpaca client has
    none, so one connection that dies silently (after a network outage, say) would freeze a
    whole job forever. With a timeout the call fails, and the callers retry."""
    global _INSTALLED
    if _INSTALLED:
        return
    original = requests.Session.request

    def request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("timeout", seconds)
        return original(self, *args, **kwargs)

    requests.Session.request = request  # type: ignore[method-assign]
    _INSTALLED = True
