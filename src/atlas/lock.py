"""Single-instance lock for long-running jobs (e.g. downloads), using a PID file."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class AlreadyRunningError(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def single_instance(name: str, root: str | Path = "data/locks") -> Iterator[None]:
    path = Path(root) / f"{name}.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        pid = int(path.read_text().strip() or 0)
        if pid and pid != os.getpid() and _alive(pid):
            raise AlreadyRunningError(f"{name} is already running (pid {pid})")
    path.write_text(str(os.getpid()))
    try:
        yield
    finally:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
