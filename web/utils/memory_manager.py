"""Memory accounting and cleanup for in-process task snapshots."""

import gc
import logging
from typing import Any

from utils.task_state_machine import RUNNING_STATES
from utils.time_utils import parse_timestamp, utc_now

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None


MAX_TASKS_IN_MEMORY = 20
TASK_RETENTION_MINUTES = 30


def cleanup_completed_task_dataframe(task: dict[str, Any]) -> bool:
    """Remove an input DataFrame once a task has completed successfully."""
    if task.get("status") == "completed" and "dataframe" in task:
        del task["dataframe"]
        return True
    return False


def cleanup_old_tasks(
    tasks_dict: dict[str, dict[str, Any]], *, keep_active: bool = True
) -> int:
    """Keep running or unsaved tasks plus the newest idle snapshots."""
    if len(tasks_dict) <= MAX_TASKS_IN_MEMORY:
        return 0

    protected_tasks = {}
    evictable_tasks = []
    cutoff = utc_now().timestamp() - TASK_RETENTION_MINUTES * 60

    for task_id, task in tasks_dict.items():
        if task.get("persistence_failed") or (
            keep_active and task.get("status") in RUNNING_STATES
        ):
            protected_tasks[task_id] = task
            continue

        timestamp = (
            task.get("completed_at")
            or task.get("failed_at")
            or task.get("cancelled_at")
            or task.get("updated_at")
            or task.get("created_at")
        )
        try:
            sort_time = parse_timestamp(timestamp).timestamp() if timestamp else 0
        except (TypeError, ValueError):
            sort_time = 0
        if sort_time >= cutoff:
            evictable_tasks.append((sort_time, task_id, task))

    evictable_tasks.sort(reverse=True)
    tasks_to_keep = dict(protected_tasks)
    remaining_slots = max(0, MAX_TASKS_IN_MEMORY - len(protected_tasks))
    for _, task_id, task in evictable_tasks[:remaining_slots]:
        tasks_to_keep[task_id] = task

    removed_count = len(tasks_dict) - len(tasks_to_keep)
    tasks_dict.clear()
    tasks_dict.update(tasks_to_keep)
    return removed_count


def get_memory_stats() -> dict[str, float]:
    """Return memory statistics for the current process."""
    if psutil is None:
        return {
            "memory_mb": 0,
            "memory_percent": 0,
            "available_system_memory_mb": 0,
        }
    try:
        process = psutil.Process()
        return {
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "memory_percent": round(process.memory_percent(), 2),
            "available_system_memory_mb": round(
                psutil.virtual_memory().available / 1024 / 1024, 2
            ),
        }
    except (psutil.Error, OSError) as error:
        logger.warning("Failed to read process memory statistics: %s", error)
        return {
            "memory_mb": 0,
            "memory_percent": 0,
            "available_system_memory_mb": 0,
        }


def estimate_dataframe_size(dataframe: Any) -> float:
    """Estimate a pandas DataFrame's deep memory usage in MB."""
    if dataframe is None or not hasattr(dataframe, "memory_usage"):
        return 0
    try:
        return float(dataframe.memory_usage(deep=True).sum()) / 1024 / 1024
    except (AttributeError, TypeError, ValueError):
        return 0


def get_tasks_memory_usage(tasks_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize task and DataFrame memory usage."""
    stats = {
        "total_tasks": len(tasks_dict),
        "tasks_with_dataframe": 0,
        "total_dataframe_size_mb": 0.0,
        "active_tasks": 0,
        "completed_tasks": 0,
    }
    for task in tasks_dict.values():
        status = task.get("status", "")
        if status in RUNNING_STATES:
            stats["active_tasks"] += 1
        elif status == "completed":
            stats["completed_tasks"] += 1
        if "dataframe" in task:
            stats["tasks_with_dataframe"] += 1
            stats["total_dataframe_size_mb"] += estimate_dataframe_size(
                task["dataframe"]
            )
    stats["total_dataframe_size_mb"] = round(stats["total_dataframe_size_mb"], 2)
    return stats


def should_trigger_cleanup(
    tasks_dict: dict[str, dict[str, Any]], memory_threshold_mb: int = 3000
) -> tuple[bool, str]:
    """Return whether task cleanup is warranted and why."""
    memory_stats = get_memory_stats()
    if memory_stats["memory_mb"] > memory_threshold_mb:
        return True, (
            f"Memory usage ({memory_stats['memory_mb']} MB) exceeds "
            f"{memory_threshold_mb} MB"
        )
    if len(tasks_dict) > MAX_TASKS_IN_MEMORY:
        return True, f"Too many tasks in memory ({len(tasks_dict)})"
    if any(
        task.get("status") == "completed" and "dataframe" in task
        for task in tasks_dict.values()
    ):
        return True, "Completed tasks still contain input DataFrames"
    return False, "No cleanup needed"


def perform_memory_cleanup(
    tasks_dict: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    """Release completed inputs, evict old tasks, and collect garbage."""
    before = get_memory_stats()
    cleaned_dataframes = sum(
        cleanup_completed_task_dataframe(task) for task in tasks_dict.values()
    )
    removed_tasks = cleanup_old_tasks(tasks_dict)
    gc.collect()
    after = get_memory_stats()
    memory_saved = round(before["memory_mb"] - after["memory_mb"], 2)
    return {
        "cleaned_dataframes": cleaned_dataframes,
        "removed_tasks": removed_tasks,
        "memory_saved_mb": memory_saved,
    }
