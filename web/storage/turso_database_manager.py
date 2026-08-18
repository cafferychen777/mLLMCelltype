#!/usr/bin/env python3
"""Turso-backed persistence for annotation tasks."""

import json
import logging
import os
from contextlib import contextmanager
from io import StringIO
from typing import Any

import libsql_experimental as libsql
import pandas as pd

from utils.serialization import to_json_compatible
from utils.time_utils import utc_now_iso

logger = logging.getLogger(__name__)


class TursoDatabaseManager:
    """Persist task snapshots in a Turso/libSQL database."""

    _COLUMN_TO_FIELD = {
        "user_ip": "ip_address",
        "file_name": "filename",
        "error_message": "error",
    }
    _METADATA_FIELDS = frozenset(
        {
            "cancelled_at",
            "cancelled_by",
            "columns",
            "completed_at",
            "consensus_model",
            "consensus_threshold",
            "entropy_threshold",
            "failed_at",
            "file_size",
            "max_rounds",
            "models",
            "owner_id",
            "owner_session_required",
            "persistence_failed",
            "preview",
            "progress",
            "queued_at",
            "retried_at",
            "retry_count",
            "run_id",
            "shape",
            "species",
            "started_at",
            "state_history",
            "state_updated_at",
            "tissue",
        }
    )
    _VALID_STATUSES = frozenset(
        {
            "cancelled",
            "completed",
            "failed",
            "file_ready",
            "processing",
            "queued",
            "timeout",
        }
    )

    def __init__(self, db_url: str | None = None, auth_token: str | None = None):
        self.db_url = db_url or os.getenv("TURSO_DB_URL")
        self.auth_token = auth_token or os.getenv("TURSO_AUTH_TOKEN")
        if not self.db_url or not self.auth_token:
            raise ValueError(
                "Missing Turso credentials. Set TURSO_DB_URL and TURSO_AUTH_TOKEN."
            )
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Yield a short-lived libSQL connection."""
        connection = libsql.connect(
            database=self.db_url,
            auth_token=self.auth_token,
        )
        try:
            yield connection
        finally:
            connection.close()

    def _init_database(self) -> None:
        """Create the schema and apply backward-compatible column migrations."""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    user_ip TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'file_ready',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    input_data TEXT,
                    results TEXT,
                    error_message TEXT,
                    cell_count INTEGER,
                    file_name TEXT,
                    last_heartbeat TIMESTAMP,
                    metadata TEXT,
                    state_version INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            cursor.execute("PRAGMA table_info(tasks)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            if "metadata" not in existing_columns:
                cursor.execute("ALTER TABLE tasks ADD COLUMN metadata TEXT")
            if "state_version" not in existing_columns:
                cursor.execute(
                    "ALTER TABLE tasks ADD COLUMN state_version INTEGER NOT NULL DEFAULT 0"
                )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at)"
            )
            connection.commit()
        logger.info("Database schema initialized")

    @classmethod
    def _serialize_metadata(cls, task_data: dict[str, Any]) -> str:
        metadata = {
            field: task_data[field]
            for field in cls._METADATA_FIELDS
            if field in task_data
        }
        return json.dumps(
            to_json_compatible(metadata),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _serialize_results(results: Any) -> str | None:
        if results is None:
            return None
        return json.dumps(
            to_json_compatible(results),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    @classmethod
    def _row_to_task(cls, columns: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
        raw = dict(zip(columns, row))
        metadata_raw = raw.pop("metadata", None)
        metadata: dict[str, Any] = {}
        if metadata_raw:
            try:
                decoded = json.loads(metadata_raw)
                if isinstance(decoded, dict):
                    metadata = decoded
            except (TypeError, json.JSONDecodeError):
                logger.warning("Ignoring malformed task metadata")

        results_raw = raw.get("results")
        if results_raw:
            try:
                raw["results"] = json.loads(results_raw)
            except (TypeError, json.JSONDecodeError):
                logger.warning("Ignoring malformed task results")
                raw["results"] = None

        # Ignore columns retained by older deployments but no longer used.
        raw.pop("processing_time_ms", None)
        raw.pop("model_used", None)

        task_data = metadata
        task_data.update(raw)
        for column, field in cls._COLUMN_TO_FIELD.items():
            if column in task_data:
                task_data[field] = task_data.pop(column)

        if task_data.get("status") == "pending":
            task_data["status"] = "queued"
        return task_data

    def save_task(self, task_id: str, task_data: dict[str, Any]) -> bool:
        """Save a coherent task snapshot without allowing stale state rollback."""
        status = task_data.get("status", "file_ready")
        if status == "pending":
            status = "queued"
        if status not in self._VALID_STATUSES:
            logger.error("Refusing to persist invalid task status %r", status)
            return False

        try:
            input_data = None
            dataframe = task_data.get("dataframe")
            if isinstance(dataframe, pd.DataFrame):
                input_data = dataframe.to_json(orient="records", date_format="iso")

            results_present = "results" in task_data
            results = self._serialize_results(task_data.get("results"))
            metadata = self._serialize_metadata(task_data)
            state_version = int(task_data.get("state_version", 0))
            now = utc_now_iso()

            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT state_version FROM tasks WHERE id = ?", (task_id,)
                )
                existing_row = cursor.fetchone()

                if existing_row and int(existing_row[0] or 0) > state_version:
                    logger.warning(
                        "Ignored stale snapshot for task %s (incoming=%s, stored=%s)",
                        task_id,
                        state_version,
                        existing_row[0],
                    )
                    return True

                values = (
                    task_data.get("ip_address") or "unknown",
                    status,
                    now,
                    input_data,
                    int(results_present),
                    results,
                    task_data.get("error"),
                    task_data.get("cell_count"),
                    task_data.get("filename"),
                    task_data.get("last_heartbeat"),
                    metadata,
                    state_version,
                )

                if existing_row:
                    cursor.execute(
                        """
                        UPDATE tasks SET
                            user_ip = ?,
                            status = ?,
                            updated_at = ?,
                            input_data = COALESCE(?, input_data),
                            results = CASE WHEN ? = 1 THEN ? ELSE results END,
                            error_message = ?,
                            cell_count = COALESCE(?, cell_count),
                            file_name = COALESCE(?, file_name),
                            last_heartbeat = ?,
                            metadata = ?,
                            state_version = ?
                        WHERE id = ? AND state_version <= ?
                        """,
                        (*values, task_id, state_version),
                    )
                    if getattr(cursor, "rowcount", 1) == 0:
                        logger.warning(
                            "Ignored concurrently superseded snapshot for task %s "
                            "(incoming=%s)",
                            task_id,
                            state_version,
                        )
                        return True
                else:
                    cursor.execute(
                        """
                        INSERT INTO tasks (
                            user_ip, status, updated_at, input_data, results,
                            error_message, cell_count, file_name, last_heartbeat, metadata,
                            state_version, id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            values[0],
                            values[1],
                            values[2],
                            values[3],
                            values[5],
                            *values[6:],
                            task_id,
                            task_data.get("created_at") or now,
                        ),
                    )
                connection.commit()

            return True
        except Exception:
            logger.exception("Failed to save task %s", task_id)
            return False

    def update_task_field(
        self,
        task_id: str,
        field: str,
        value: Any,
        *,
        expected_state_version: int | None = None,
    ) -> bool:
        """Persist a heartbeat without serializing the full task snapshot."""
        if field != "last_heartbeat":
            logger.error("Only last_heartbeat supports a direct update")
            return False
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                if expected_state_version is None:
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET last_heartbeat = ?, updated_at = ?
                        WHERE id = ? AND status = 'processing'
                        """,
                        (value, utc_now_iso(), task_id),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET last_heartbeat = ?, updated_at = ?
                        WHERE id = ?
                          AND status = 'processing'
                          AND state_version = ?
                        """,
                        (value, utc_now_iso(), task_id, expected_state_version),
                    )
                connection.commit()
                return cursor.rowcount > 0
        except Exception:
            logger.exception("Failed to update heartbeat for task %s", task_id)
            return False

    def get_task(
        self, task_id: str, include_dataframe: bool = False
    ) -> dict[str, Any] | None:
        """Load one task, optionally reconstructing its uploaded DataFrame."""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                columns = [description[0] for description in cursor.description]

            task_data = self._row_to_task(columns, row)
            input_data = task_data.pop("input_data", None)
            if include_dataframe and input_data:
                try:
                    task_data["dataframe"] = pd.read_json(
                        StringIO(input_data), orient="records"
                    )
                except (TypeError, ValueError):
                    logger.exception(
                        "Failed to reconstruct DataFrame for task %s", task_id
                    )
            return task_data
        except Exception:
            logger.exception("Failed to load task %s", task_id)
            return None

    def get_all_tasks(self) -> dict[str, dict[str, Any]]:
        """Load lightweight task history without input or result payloads."""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    SELECT id, user_ip, status, created_at, updated_at,
                           error_message, cell_count, file_name, last_heartbeat, metadata,
                           state_version
                    FROM tasks
                    ORDER BY created_at DESC
                    """
                )
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            return {
                task["id"]: task
                for task in (self._row_to_task(columns, row) for row in rows)
            }
        except Exception:
            logger.exception("Failed to load task history")
            return {}

    def delete_task(self, task_id: str) -> bool:
        """Delete one task."""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                connection.commit()
                return cursor.rowcount > 0
        except Exception:
            logger.exception("Failed to delete task %s", task_id)
            return False

    def load_active_tasks(self) -> dict[str, dict[str, Any]]:
        """Load lightweight tasks that require startup recovery or monitoring."""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    SELECT id, user_ip, status, created_at, updated_at,
                           error_message, cell_count, file_name, last_heartbeat, metadata,
                           state_version
                    FROM tasks
                    WHERE status IN ('processing', 'queued', 'pending')
                    ORDER BY updated_at DESC
                    """
                )
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            return {
                task["id"]: task
                for task in (self._row_to_task(columns, row) for row in rows)
            }
        except Exception:
            logger.exception("Failed to load active tasks")
            return {}
