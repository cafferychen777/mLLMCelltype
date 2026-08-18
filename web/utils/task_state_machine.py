"""Validated task-state transitions for mLLMCelltype Web."""

from enum import Enum
from typing import Any, ClassVar

from utils.time_utils import parse_timestamp, utc_now, utc_now_iso


class TaskState(Enum):
    """Valid task states."""

    FILE_READY = "file_ready"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


TERMINAL_STATES = frozenset(
    {
        TaskState.COMPLETED.value,
        TaskState.FAILED.value,
        TaskState.CANCELLED.value,
        TaskState.TIMEOUT.value,
    }
)
RUNNING_STATES = frozenset(
    {
        TaskState.QUEUED.value,
        TaskState.PROCESSING.value,
    }
)
CANCELLABLE_STATES = frozenset(
    {
        TaskState.FILE_READY.value,
        *RUNNING_STATES,
    }
)


class TaskStateError(ValueError):
    """Raised when a task has an invalid state or transition."""


class TaskStateMachine:
    """Apply state transitions to a task dictionary.

    The caller must hold the lock that protects the shared task dictionary.
    A lock owned by each state-machine instance would provide false safety
    because the application creates a new instance for every transition.
    """

    VALID_TRANSITIONS: ClassVar[dict[TaskState, frozenset[TaskState]]] = {
        TaskState.FILE_READY: frozenset({TaskState.QUEUED, TaskState.CANCELLED}),
        TaskState.QUEUED: frozenset(
            {TaskState.PROCESSING, TaskState.FAILED, TaskState.CANCELLED}
        ),
        TaskState.PROCESSING: frozenset(
            {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.TIMEOUT,
                TaskState.CANCELLED,
            }
        ),
        TaskState.COMPLETED: frozenset({TaskState.FILE_READY}),
        TaskState.FAILED: frozenset({TaskState.FILE_READY}),
        TaskState.CANCELLED: frozenset({TaskState.FILE_READY}),
        TaskState.TIMEOUT: frozenset({TaskState.FILE_READY}),
    }

    _RUN_FIELDS: ClassVar[set[str]] = {
        "cancelled_at",
        "cancelled_by",
        "completed_at",
        "error",
        "failed_at",
        "last_heartbeat",
        "persistence_failed",
        "progress",
        "queued_at",
        "run_id",
        "started_at",
        "worker_thread_id",
    }
    _MAX_HISTORY_ENTRIES = 100

    def __init__(self, task_dict: dict[str, Any]):
        self.task = task_dict

    def get_state(self) -> TaskState:
        """Return the current state, normalizing the legacy pending value."""
        status = self.task.get("status", TaskState.FILE_READY.value)
        if status == "pending":
            status = TaskState.QUEUED.value
            self.task["status"] = status
        try:
            return TaskState(status)
        except ValueError as exc:
            raise TaskStateError(f"Unknown task status: {status!r}") from exc

    def can_transition_to(self, new_state: TaskState) -> bool:
        """Return whether the requested transition is valid."""
        return new_state in self.VALID_TRANSITIONS[self.get_state()]

    def transition_to(self, new_state: TaskState, error: str | None = None) -> bool:
        """Transition to ``new_state`` and update state-specific metadata."""
        if not isinstance(new_state, TaskState):
            raise TaskStateError(f"Invalid target state: {new_state!r}")

        current_state = self.get_state()
        if new_state not in self.VALID_TRANSITIONS[current_state]:
            raise TaskStateError(
                f"Invalid transition from {current_state.value} to {new_state.value}"
            )

        timestamp = utc_now_iso()
        self.task["status"] = new_state.value
        self.task["state_updated_at"] = timestamp
        self.task["state_version"] = int(self.task.get("state_version", 0)) + 1
        state_history = self.task.setdefault("state_history", [])
        state_history.append(
            {
                "from": current_state.value,
                "to": new_state.value,
                "timestamp": timestamp,
                "error": error,
            }
        )
        if len(state_history) > self._MAX_HISTORY_ENTRIES:
            del state_history[: -self._MAX_HISTORY_ENTRIES]

        if new_state == TaskState.FILE_READY:
            for field in self._RUN_FIELDS:
                self.task.pop(field, None)
            # An explicit None tells the persistence layer to clear stale output.
            self.task["results"] = None
        elif new_state == TaskState.QUEUED:
            for field in self._RUN_FIELDS:
                self.task.pop(field, None)
            self.task["results"] = None
            self.task["queued_at"] = timestamp
        elif new_state == TaskState.PROCESSING:
            for field in ("cancelled_at", "completed_at", "failed_at", "error"):
                self.task.pop(field, None)
            self.task["started_at"] = timestamp
            self.task["last_heartbeat"] = timestamp
        elif new_state == TaskState.COMPLETED:
            for field in ("cancelled_at", "failed_at", "error"):
                self.task.pop(field, None)
            self.task["completed_at"] = timestamp
        elif new_state in {TaskState.FAILED, TaskState.TIMEOUT}:
            self.task.pop("completed_at", None)
            self.task.pop("cancelled_at", None)
            self.task["failed_at"] = timestamp
            if error:
                self.task["error"] = error
        elif new_state == TaskState.CANCELLED:
            self.task.pop("completed_at", None)
            self.task.pop("failed_at", None)
            self.task["cancelled_at"] = timestamp
            if error:
                self.task["error"] = error

        return True

    def check_timeout(self, timeout_seconds: int = 600) -> bool:
        """Transition a processing task to timeout after a stale heartbeat."""
        if self.get_state() != TaskState.PROCESSING:
            return False

        last_heartbeat = self.task.get("last_heartbeat")
        if not last_heartbeat:
            return False

        if (
            utc_now() - parse_timestamp(last_heartbeat)
        ).total_seconds() <= timeout_seconds:
            return False

        self.transition_to(
            TaskState.TIMEOUT,
            f"Task timed out after {timeout_seconds} seconds without heartbeat",
        )
        return True
