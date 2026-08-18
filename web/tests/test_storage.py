"""Tests for persistence metadata compatibility."""

import json
import math
from contextlib import contextmanager

import pandas as pd

from storage.turso_database_manager import TursoDatabaseManager


class FakeCursor:
    def __init__(self, existing=None):
        self.existing = existing
        self.executions = []

    def execute(self, query, parameters=()):
        self.executions.append((" ".join(query.split()), parameters))

    def fetchone(self):
        return self.existing


class FakeConnection:
    def __init__(self, existing=None):
        self.cursor_instance = FakeCursor(existing)
        self.committed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True


def test_metadata_round_trip_preserves_task_configuration() -> None:
    metadata = TursoDatabaseManager._serialize_metadata(
        {
            "species": "human",
            "tissue": "blood",
            "models": ["openai:gpt-4.1"],
            "progress": {"percentage": math.nan},
            "ignored_secret": "do-not-store",
        }
    )

    decoded = json.loads(metadata)
    assert decoded["species"] == "human"
    assert decoded["models"] == ["openai:gpt-4.1"]
    assert decoded["progress"]["percentage"] is None
    assert "ignored_secret" not in decoded


def test_row_conversion_merges_metadata_and_canonical_columns() -> None:
    task = TursoDatabaseManager._row_to_task(
        [
            "id",
            "user_ip",
            "status",
            "file_name",
            "metadata",
            "results",
            "state_version",
            "processing_time_ms",
            "model_used",
        ],
        (
            "task-1",
            "127.0.0.1",
            "pending",
            "markers.csv",
            '{"species":"mouse","progress":{"percentage":40}}',
            '{"consensus":{"0":"T cell"}}',
            2,
            1200,
            "legacy-model",
        ),
    )

    assert task["status"] == "queued"
    assert task["ip_address"] == "127.0.0.1"
    assert task["filename"] == "markers.csv"
    assert task["species"] == "mouse"
    assert task["results"]["consensus"]["0"] == "T cell"
    assert "processing_time_ms" not in task
    assert "model_used" not in task


def manager_with_connection(connection):
    manager = TursoDatabaseManager.__new__(TursoDatabaseManager)

    @contextmanager
    def get_connection():
        yield connection

    manager.get_connection = get_connection
    return manager


def test_insert_statement_receives_one_value_per_column() -> None:
    connection = FakeConnection()
    manager = manager_with_connection(connection)

    assert manager.save_task(
        "task-1",
        {
            "status": "file_ready",
            "state_version": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "ip_address": "127.0.0.1",
            "filename": "markers.csv",
        },
    )

    query, parameters = connection.cursor_instance.executions[-1]
    assert query.startswith("INSERT INTO tasks")
    assert query.count("?") == len(parameters) == 13
    assert connection.committed


def test_update_can_explicitly_clear_stale_results() -> None:
    connection = FakeConnection(existing=(2,))
    manager = manager_with_connection(connection)

    assert manager.save_task(
        "task-1",
        {
            "status": "file_ready",
            "state_version": 3,
            "results": None,
            "ip_address": "127.0.0.1",
        },
    )

    query, parameters = connection.cursor_instance.executions[-1]
    assert query.startswith("UPDATE tasks SET")
    assert "CASE WHEN ? = 1 THEN ? ELSE results END" in query
    assert parameters[4:6] == (1, None)


def test_real_libsql_round_trip_and_stale_write_protection(tmp_path) -> None:
    manager = TursoDatabaseManager(
        f"file:{tmp_path / 'tasks.db'}",
        "local-test-token",
    )
    task_id = "task-1"
    task = {
        "status": "file_ready",
        "state_version": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "ip_address": "127.0.0.1",
        "filename": "markers.csv",
        "owner_id": "owner-" + "x" * 32,
        "dataframe": pd.DataFrame({"Cluster 0": ["CD3D", "CD3E"]}),
    }

    assert manager.save_task(task_id, task)
    loaded = manager.get_task(task_id, include_dataframe=True)
    assert loaded is not None
    assert loaded["owner_id"] == task["owner_id"]
    assert loaded["dataframe"].to_dict("list") == {"Cluster 0": ["CD3D", "CD3E"]}

    task.update(
        {
            "status": "completed",
            "state_version": 1,
            "results": {"consensus": {"0": "T cell"}},
        }
    )
    assert manager.save_task(task_id, task)
    assert manager.get_task(task_id)["results"]["consensus"]["0"] == "T cell"

    task.update({"status": "file_ready", "state_version": 2, "results": None})
    assert manager.save_task(task_id, task)
    assert manager.get_task(task_id)["results"] is None

    stale = dict(
        task,
        status="completed",
        state_version=1,
        results={"stale": True},
    )
    assert manager.save_task(task_id, stale)
    assert manager.get_task(task_id)["status"] == "file_ready"
