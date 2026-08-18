"""Tests for task lifecycle invariants."""

from datetime import timedelta

import pytest

from utils.task_state_machine import TaskState, TaskStateError, TaskStateMachine
from utils.time_utils import utc_now


def test_reset_clears_previous_run_state() -> None:
    task = {
        "status": "completed",
        "state_version": 4,
        "results": {"consensus": {"0": "T cell"}},
        "completed_at": "2026-01-01T00:00:00+00:00",
        "progress": {"percentage": 100},
        "error": "stale",
    }

    TaskStateMachine(task).transition_to(TaskState.FILE_READY)

    assert task["status"] == "file_ready"
    assert task["state_version"] == 5
    assert task["results"] is None
    assert "completed_at" not in task
    assert "progress" not in task
    assert "error" not in task


def test_unknown_state_is_rejected() -> None:
    with pytest.raises(TaskStateError, match="Unknown task status"):
        TaskStateMachine({"status": "mystery"}).get_state()


def test_stale_processing_task_times_out() -> None:
    task = {
        "status": "processing",
        "last_heartbeat": (utc_now() - timedelta(minutes=20)).isoformat(),
    }

    assert TaskStateMachine(task).check_timeout(timeout_seconds=60)
    assert task["status"] == "timeout"
    assert "timed out" in task["error"]


def test_invalid_transition_does_not_mutate_task() -> None:
    task = {"status": "file_ready"}

    with pytest.raises(TaskStateError, match="Invalid transition"):
        TaskStateMachine(task).transition_to(TaskState.COMPLETED)

    assert task == {"status": "file_ready"}
