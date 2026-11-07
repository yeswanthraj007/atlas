import requests

from atlas.net import install_default_timeout


def test_default_timeout_is_added_but_never_overrides(monkeypatch):
    seen = {}

    def fake(self, *args, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(requests.Session, "request", fake)
    monkeypatch.setattr("atlas.net._INSTALLED", False)
    install_default_timeout(12.0)
    s = requests.Session()
    s.request("GET", "http://example.invalid")
    assert seen["timeout"] == 12.0
    s.request("GET", "http://example.invalid", timeout=3)
    assert seen["timeout"] == 3


def test_with_retries_recovers_from_dropped_connections_then_gives_up():
    import pytest

    from atlas.recommend.daily import with_retries

    waits, calls = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ConnectionError("remote end closed connection")
        return "ok"

    assert with_retries(flaky, sleep=waits.append) == "ok"
    assert waits == [20.0, 40.0] and len(calls) == 3

    def dead():
        raise requests.exceptions.ConnectionError("down")

    with pytest.raises(requests.exceptions.ConnectionError):
        with_retries(dead, tries=2, sleep=lambda _: None)
