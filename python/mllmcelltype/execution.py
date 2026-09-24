"""Run-scoped execution limits and progress shared by the annotation pipeline."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


class AnnotationStopped(Exception):
    """A caller cancelled the run or its execution budget was exhausted."""


@dataclass
class AnnotationExecution:
    """One run's state; never shared between concurrent annotation jobs."""

    request_timeout: float
    max_runtime: float
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    should_cancel: Callable[[], bool] | None = None
    started: float = field(default_factory=time.monotonic)
    phase: str = "annotation"
    cluster: str | None = None
    round: int | None = None
    unavailable: dict[str, str] = field(default_factory=dict)
    completed_calls: int = 0
    use_cache: bool = True
    consensus_candidates: list[tuple[str, str]] = field(default_factory=list)

    def remaining(self) -> float:
        if self.should_cancel is not None and self.should_cancel():
            raise AnnotationStopped("Annotation cancelled")
        remaining = self.max_runtime - (time.monotonic() - self.started)
        if remaining <= 0:
            raise AnnotationStopped("Annotation exceeded its execution time budget")
        return remaining

    def emit(self, event: str, **details: Any) -> None:
        self.remaining()
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "event": event,
                    "phase": self.phase,
                    "cluster": self.cluster,
                    "round": self.round,
                    "elapsed_seconds": round(time.monotonic() - self.started, 1),
                    "completed_calls": self.completed_calls,
                    "unavailable_models": dict(self.unavailable),
                    **details,
                }
            )


_execution: contextvars.ContextVar[AnnotationExecution | None] = contextvars.ContextVar(
    "annotation_execution", default=None
)


def current_execution() -> AnnotationExecution | None:
    return _execution.get()


@contextlib.contextmanager
def annotation_execution(
    *,
    request_timeout: float,
    max_runtime: float,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> Iterator[AnnotationExecution]:
    for name, value in (("request_timeout", request_timeout), ("max_runtime", max_runtime)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number")
    for name, value in (("progress_callback", progress_callback), ("should_cancel", should_cancel)):
        if value is not None and not callable(value):
            raise ValueError(f"{name} must be callable or None")
    run = AnnotationExecution(request_timeout, max_runtime, progress_callback, should_cancel)
    token = _execution.set(run)
    try:
        yield run
    finally:
        _execution.reset(token)


def track_model_call(function):
    """Observe provider calls, including cache hits, at their common boundary."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        run = current_execution()
        if run is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        provider = bound.arguments.get("provider")
        model = bound.arguments.get("model")
        key = f"{provider}:{model}"
        run.remaining()
        if key in run.unavailable:
            raise RuntimeError(f"Model {key} is unavailable for this run")
        run.emit("model_started", model=key)
        try:
            result = function(*args, **kwargs)
        except AnnotationStopped:
            raise
        except Exception as error:
            # Public telemetry contains a category, never provider bodies or credentials.
            run.unavailable[key] = type(error).__name__
            run.emit("model_failed", model=key, error=type(error).__name__)
            raise
        run.completed_calls += 1
        run.emit("model_completed", model=key)
        return result

    return wrapped
