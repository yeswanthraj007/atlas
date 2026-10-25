import os

import pytest

from atlas.lock import AlreadyRunningError, single_instance


def test_second_instance_is_refused_and_stale_locks_are_cleared(tmp_path) -> None:
    with single_instance("job", tmp_path):
        (tmp_path / "other.pid").write_text(str(os.getppid()))  # a live foreign process
        with pytest.raises(AlreadyRunningError), single_instance("other", tmp_path):
            pass
    assert not (tmp_path / "job.pid").exists()
    (tmp_path / "stale.pid").write_text("999999")  # process that no longer exists
    with single_instance("stale", tmp_path):
        pass
