"""Tests for task snapshot memory cleanup decisions."""

import pandas as pd

from utils.memory_manager import should_trigger_cleanup


def test_cleanup_requires_dataframe_on_the_completed_task(monkeypatch) -> None:
    monkeypatch.setattr(
        "utils.memory_manager.get_memory_stats",
        lambda: {
            "memory_mb": 100,
            "memory_percent": 1,
            "available_system_memory_mb": 1000,
        },
    )
    tasks = {
        "completed": {"status": "completed"},
        "active": {
            "status": "processing",
            "dataframe": pd.DataFrame({"gene": ["CD3D"]}),
        },
    }

    assert should_trigger_cleanup(tasks) == (False, "No cleanup needed")


def test_cleanup_detects_completed_dataframe(monkeypatch) -> None:
    monkeypatch.setattr(
        "utils.memory_manager.get_memory_stats",
        lambda: {
            "memory_mb": 100,
            "memory_percent": 1,
            "available_system_memory_mb": 1000,
        },
    )
    tasks = {
        "completed": {
            "status": "completed",
            "dataframe": pd.DataFrame({"gene": ["CD3D"]}),
        }
    }

    should_cleanup, reason = should_trigger_cleanup(tasks)
    assert should_cleanup is True
    assert reason == "Completed tasks still contain input DataFrames"
