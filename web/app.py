#!/usr/bin/env python3
"""
Simple, reliable backend for mLLMCelltype Web
Rewritten for clarity and robustness
"""

import io
import importlib.metadata
import json
import logging
import math
import os
import secrets
import time
import uuid
import copy
import threading
import atexit
from datetime import timedelta
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    has_request_context,
    request,
    jsonify,
    render_template,
    send_file,
    session,
    redirect,
    Response,
    url_for,
)
from flask.sessions import SecureCookieSessionInterface
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash

from config.model_catalog import (
    MODEL_CATALOG,
    MODEL_CATALOG_UPDATED_AT,
    get_provider_defaults,
    get_provider_names,
    get_serialized_catalog,
)
from storage.turso_database_manager import TursoDatabaseManager
from utils.marker_file import MarkerFileError, get_upload_size, read_marker_dataframe
from utils.request_context import get_client_ip
from utils.serialization import to_json_compatible
from utils.task_state_machine import (
    CANCELLABLE_STATES,
    RUNNING_STATES,
    TERMINAL_STATES,
    TaskState,
    TaskStateError,
    TaskStateMachine,
)
from utils.task_validation import (
    MAX_API_KEY_LENGTH,
    SUPPORTED_PROVIDERS,
    AnnotationValidationError,
    parse_annotation_request,
)
from utils.time_utils import parse_timestamp, utc_now, utc_now_iso

load_dotenv()

logger = logging.getLogger(__name__)

# Capture process start time once as a stable deployment-time fallback.
_APP_START_TIME = utc_now_iso()

try:
    _ANNOTATION_ENGINE_VERSION = importlib.metadata.version("mllmcelltype")
except importlib.metadata.PackageNotFoundError:
    _ANNOTATION_ENGINE_VERSION = "unknown"


class _SchemeAwareSessionInterface(SecureCookieSessionInterface):
    """Set the session cookie Secure flag based on the actual request scheme.

    First principles:
    - On HTTPS requests (including behind ProxyFix), the cookie must be Secure.
    - On local HTTP dev, Secure cookies would not be sent, breaking login state.
    """

    def get_cookie_secure(self, app: Flask) -> bool:  # type: ignore[override]
        # Allow an explicit, global override via config if ever needed.
        if bool(app.config.get("SESSION_COOKIE_SECURE")):
            return True
        if has_request_context():
            return bool(request.is_secure)
        return False


try:
    from utils.memory_manager import (
        cleanup_completed_task_dataframe,
        get_memory_stats,
        get_tasks_memory_usage,
        should_trigger_cleanup,
        perform_memory_cleanup,
    )

    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    logger.warning("Memory management utilities are unavailable")
    MEMORY_MANAGEMENT_AVAILABLE = False

LOCAL_TESTING = os.getenv("LOCAL_TESTING", "false").lower() == "true"
BACKGROUND_THREADS_ENABLED = (
    os.getenv("BACKGROUND_THREADS_ENABLED", str(not LOCAL_TESTING)).lower() == "true"
)

if LOCAL_TESTING:

    class MockStorageManager:
        """In-memory storage that mirrors the production storage interface."""

        def __init__(self):
            self.tasks = {}

        def save_task(self, task_id, task_data):
            self.tasks[task_id] = copy.deepcopy(task_data)
            return True

        def get_task(self, task_id, include_dataframe=False):
            task = self.tasks.get(task_id)
            if task is None:
                return None
            result = copy.deepcopy(task)
            if not include_dataframe:
                result.pop("dataframe", None)
            return result

        def get_all_tasks(self):
            return {
                task_id: {
                    key: copy.deepcopy(value)
                    for key, value in task.items()
                    if key not in {"dataframe", "results"}
                }
                for task_id, task in self.tasks.items()
            }

        def load_active_tasks(self):
            return {
                task_id: copy.deepcopy(task)
                for task_id, task in self.tasks.items()
                if task.get("status") in {"queued", "processing"}
            }

        def update_task_field(
            self, task_id, field, value, *, expected_state_version=None
        ):
            if task_id not in self.tasks or field != "last_heartbeat":
                return False
            task = self.tasks[task_id]
            if task.get("status") != TaskState.PROCESSING.value:
                return False
            if (
                expected_state_version is not None
                and task.get("state_version", 0) != expected_state_version
            ):
                return False
            self.tasks[task_id][field] = value
            return True

        def delete_task(self, task_id):
            return self.tasks.pop(task_id, None) is not None

    storage_manager = MockStorageManager()
    logger.info("Using in-memory storage")
else:
    storage_manager = TursoDatabaseManager()
    logger.info("Using Turso storage")

if hasattr(storage_manager, "db_url"):
    _db_host = urlsplit(storage_manager.db_url).hostname or "local"
    logger.info("Database host: %s", _db_host)

# In-memory task store — authoritative during process lifetime.
# DB (Turso) is the persistence layer, written asynchronously.
TASKS = {}
TASKS_LOCK = threading.RLock()

# Concurrency limit for annotation worker threads
MAX_CONCURRENT_ANNOTATIONS = 5
_annotation_slots = threading.Semaphore(MAX_CONCURRENT_ANNOTATIONS)

# Heartbeat configuration
HEARTBEAT_TIMEOUT_SECONDS = 900  # 15 minutes - task considered stale if no heartbeat
QUEUED_TIMEOUT_SECONDS = (
    120  # 2 minutes - queued task considered stuck if never started
)
HEARTBEAT_BUFFER = {}
HEARTBEAT_BUFFER_LOCK = threading.Lock()

# Admin credentials — no defaults; admin is disabled unless explicitly configured.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")
ADMIN_ENABLED = bool(ADMIN_USERNAME and ADMIN_PASSWORD_HASH)

if not ADMIN_ENABLED:
    logger.info("Admin dashboard is disabled because credentials are not configured")

_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _sanitize_for_csv(value: str) -> str:
    """Prevent CSV formula injection (OWASP).

    If a cell value starts with =, +, -, @, tab, or CR, Excel/Sheets may
    interpret it as a formula.  Prefixing with a single-quote neutralises
    the threat while remaining human-readable.
    """
    if value and value[0] in _CSV_FORMULA_PREFIXES:
        return "'" + value
    return value


def _state_history_logs(history):
    """Convert persisted state transitions into the admin log contract."""
    if not isinstance(history, list):
        return []

    logs = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        previous_state = str(entry.get("from", "unknown"))
        next_state = str(entry.get("to", "unknown"))
        error = str(entry.get("error") or "").strip()
        if next_state in {TaskState.FAILED.value, TaskState.TIMEOUT.value}:
            level = "error"
        elif next_state == TaskState.CANCELLED.value:
            level = "warning"
        else:
            level = "info"
        message = f"State changed from {previous_state} to {next_state}"
        if error:
            message = f"{message}: {error}"
        logs.append(
            {
                "timestamp": entry.get("timestamp", ""),
                "level": level,
                "message": message,
            }
        )
    return logs


class MemoryCleanupThread:
    """Periodically clean up in-memory task payloads."""

    def __init__(self, interval=300):  # Clean every 5 minutes
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start the memory cleanup thread when enabled."""
        if MEMORY_MANAGEMENT_AVAILABLE and BACKGROUND_THREADS_ENABLED:
            self._thread = threading.Thread(
                target=self._run, name="memory-cleanup", daemon=True
            )
            self._thread.start()
            logger.info("Started memory cleanup thread (interval=%ss)", self.interval)

    def stop(self):
        """Stop the memory cleanup thread"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=min(self.interval, 5))
        logger.info("Stopped memory cleanup thread")

    def _run(self):
        """Run the memory cleanup loop"""
        while not self._stop_event.is_set():
            try:
                recovered_saves = retry_failed_task_saves()
                if recovered_saves:
                    logger.info(
                        "Recovered persistence for %s task snapshots",
                        recovered_saves,
                    )

                # Check if cleanup is needed (acquire lock for TASKS access)
                with TASKS_LOCK:
                    should_cleanup, reason = should_trigger_cleanup(TASKS)
                if should_cleanup:
                    logger.info("Memory cleanup triggered: %s", reason)
                    with TASKS_LOCK:
                        cleanup_results = perform_memory_cleanup(TASKS)
                    logger.info("Memory cleanup results: %s", cleanup_results)
                else:
                    # Just log memory stats
                    stats = get_memory_stats()
                    with TASKS_LOCK:
                        task_stats = get_tasks_memory_usage(TASKS)
                    logger.info(
                        "Memory stats: %s MB, %s tasks, %s DataFrames",
                        stats["memory_mb"],
                        task_stats["total_tasks"],
                        task_stats["tasks_with_dataframe"],
                    )
            except Exception as e:
                logger.exception("Memory cleanup failed: %s", e)

            # Wait for next cleanup
            self._stop_event.wait(self.interval)


def flush_heartbeat_buffer():
    """Flush heartbeat buffer to database.

    Lock ordering: TASKS_LOCK -> HEARTBEAT_BUFFER_LOCK to prevent deadlocks.
    I/O (update_task_field) is done outside both locks — lightweight
    single-column update, no DataFrame serialization.
    """
    # Snapshot buffer under HEARTBEAT_BUFFER_LOCK only
    with HEARTBEAT_BUFFER_LOCK:
        if not HEARTBEAT_BUFFER:
            return
        buffer_snapshot = dict(HEARTBEAT_BUFFER)

    # Apply heartbeats to TASKS under TASKS_LOCK, collect save payloads
    save_payloads = {}
    orphaned_ids = []
    with TASKS_LOCK:
        for task_id, (run_id, state_version, heartbeat) in buffer_snapshot.items():
            task = TASKS.get(task_id)
            if (
                task is not None
                and task.get("status") == TaskState.PROCESSING.value
                and task.get("run_id") == run_id
            ):
                task["last_heartbeat"] = heartbeat
                save_payloads[task_id] = (heartbeat, state_version)
            else:
                # The task disappeared or advanced to another run.
                orphaned_ids.append(task_id)

    # I/O outside all locks — lightweight single-column update
    saved_ids = []
    try:
        for task_id, (heartbeat, state_version) in save_payloads.items():
            if storage_manager.update_task_field(
                task_id,
                "last_heartbeat",
                heartbeat,
                expected_state_version=state_version,
            ):
                saved_ids.append(task_id)
            else:
                logger.debug("Heartbeat was not applied for task %s", task_id)
    except Exception as e:
        logger.exception("Heartbeat buffer flush failed: %s", e)

    # Clear flushed + orphaned entries from buffer.
    # Only pop if value hasn't changed since snapshot (a newer heartbeat
    # may have been written concurrently; keep it for the next flush).
    with HEARTBEAT_BUFFER_LOCK:
        for task_id in saved_ids:
            if HEARTBEAT_BUFFER.get(task_id) == buffer_snapshot.get(task_id):
                del HEARTBEAT_BUFFER[task_id]
        for task_id in orphaned_ids:
            if HEARTBEAT_BUFFER.get(task_id) == buffer_snapshot.get(task_id):
                del HEARTBEAT_BUFFER[task_id]


class TaskMonitoringThread:
    """Monitor active tasks and transition stale tasks to terminal states."""

    def __init__(self, interval=60):  # Check every 60 seconds
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start the task monitoring thread when enabled."""
        if not BACKGROUND_THREADS_ENABLED:
            return
        self._thread = threading.Thread(
            target=self._run, name="task-monitor", daemon=True
        )
        self._thread.start()
        logger.info("Started task monitoring thread (interval=%ss)", self.interval)

    def stop(self):
        """Stop the task monitoring thread"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=min(self.interval, 5))
        logger.info("Stopped task monitoring thread")

    def _run(self):
        """Run the task monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Load only active tasks from storage (processing and queued)
                load_active_tasks()

                # Snapshot task IDs to check (not mutable references)
                with TASKS_LOCK:
                    candidates = [
                        tid
                        for tid, task in TASKS.items()
                        if task.get("status") in RUNNING_STATES
                    ]

                stuck_tasks = []
                recovered_tasks = []

                for task_id in candidates:
                    try:
                        with TASKS_LOCK:
                            if task_id not in TASKS:
                                continue
                            task = TASKS[task_id]
                            state_machine = TaskStateMachine(task)

                            # Check if task has timed out
                            if state_machine.check_timeout(
                                timeout_seconds=HEARTBEAT_TIMEOUT_SECONDS
                            ):
                                stuck_tasks.append(task_id)

                            # Check for tasks stuck in queued state
                            if state_machine.get_state() == TaskState.QUEUED:
                                queued_at = task.get("queued_at")
                                if queued_at:
                                    try:
                                        queued_time = parse_timestamp(queued_at)
                                        if (
                                            utc_now() - queued_time
                                        ).total_seconds() > QUEUED_TIMEOUT_SECONDS:
                                            state_machine.transition_to(
                                                TaskState.FAILED,
                                                "Task stuck in queued state for too long",
                                            )
                                            recovered_tasks.append(task_id)
                                    except Exception as e:
                                        logger.warning(
                                            "Failed to inspect queued task %s: %s",
                                            task_id,
                                            e,
                                        )
                    except Exception as e:
                        # Skip tasks that can't be processed by state machine
                        # This handles legacy tasks with invalid states
                        logger.warning("Failed to monitor task %s: %s", task_id, e)

                    # Save outside lock for each affected task
                    if task_id in stuck_tasks or task_id in recovered_tasks:
                        _bg_save(task_id, "monitor_recovery")

                if stuck_tasks or recovered_tasks:
                    logger.warning(
                        "Task monitor timed out %s tasks and recovered %s queued tasks",
                        len(stuck_tasks),
                        len(recovered_tasks),
                    )

            except Exception as e:
                logger.exception("Task monitor failed: %s", e)

            # Wait for next check
            self._stop_event.wait(self.interval)


class HeartbeatUpdater:
    """Thread to periodically update task heartbeat"""

    def __init__(self, task_id, run_id, interval=30):
        self.task_id = task_id
        self.run_id = run_id
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start the heartbeat updater thread"""
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{self.task_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started heartbeat updater for task %s", self.task_id)

    def stop(self):
        """Stop the heartbeat updater"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=min(self.interval, 5))
        logger.info("Stopped heartbeat updater for task %s", self.task_id)

    def _run(self):
        """Run the heartbeat update loop"""
        last_flush = time.time()  # Track last flush time

        while not self._stop_event.is_set():
            try:
                # Check task status under lock
                with TASKS_LOCK:
                    task = TASKS.get(self.task_id)
                    is_processing = bool(
                        task
                        and task.get("status") == TaskState.PROCESSING.value
                        and task.get("run_id") == self.run_id
                    )
                    state_version = task.get("state_version", 0) if task else 0

                if is_processing:
                    # Directly update buffer without triggering flush to avoid blocking
                    with HEARTBEAT_BUFFER_LOCK:
                        HEARTBEAT_BUFFER[self.task_id] = (
                            self.run_id,
                            state_version,
                            utc_now_iso(),
                        )

                    # Force flush to database every 60 seconds to prevent timeout issues
                    if time.time() - last_flush > 60:
                        try:
                            flush_heartbeat_buffer()
                            last_flush = time.time()
                        except Exception as flush_error:
                            logger.warning(
                                "Forced heartbeat flush failed for task %s: %s",
                                self.task_id,
                                flush_error,
                            )
                else:
                    break
            except Exception as e:
                logger.exception(
                    "Heartbeat update failed for task %s: %s", self.task_id, e
                )

            # Wait for next update
            self._stop_event.wait(self.interval)


def _copy_task_for_persistence(source):
    """Copy mutable task fields while retaining the read-only DataFrame."""
    return {
        key: value if key == "dataframe" else copy.deepcopy(value)
        for key, value in source.items()
    }


def save_single_task(task_id, reason="manual"):
    """Snapshot and persist one task without holding the shared lock for I/O."""
    try:
        with TASKS_LOCK:
            if task_id not in TASKS:
                logger.warning("Task %s was not found for save (%s)", task_id, reason)
                return False
            task_data = _copy_task_for_persistence(TASKS[task_id])

        success = storage_manager.save_task(task_id, task_data)
        if not success:
            logger.error("Failed to save task %s (%s)", task_id, reason)
        return success
    except Exception:
        logger.exception("Failed to save task %s (%s)", task_id, reason)
        return False


def retry_failed_task_saves():
    """Retry terminal snapshots that survived a transient storage failure."""
    with TASKS_LOCK:
        pending = {}
        for task_id, task in TASKS.items():
            if not task.get("persistence_failed"):
                continue
            snapshot = _copy_task_for_persistence(task)
            snapshot.pop("persistence_failed", None)
            pending[task_id] = (
                snapshot,
                task.get("status"),
                task.get("state_version", 0),
                task.get("run_id"),
            )

    recovered = 0
    for task_id, (snapshot, status, state_version, run_id) in pending.items():
        try:
            saved = storage_manager.save_task(task_id, snapshot)
        except Exception:
            logger.exception("Persistence retry failed for task %s", task_id)
            continue
        if not saved:
            continue

        with TASKS_LOCK:
            current = TASKS.get(task_id)
            if (
                current is not None
                and current.get("status") == status
                and current.get("state_version", 0) == state_version
                and current.get("run_id") == run_id
            ):
                current.pop("persistence_failed", None)
                recovered += 1

    return recovered


def save_tasks():
    """Bulk save all in-memory tasks to database.

    Only used by the atexit handler to persist final state on shutdown.
    For single-task operations, use save_single_task() instead.
    """
    try:
        with TASKS_LOCK:
            tasks_snapshot = {
                task_id: _copy_task_for_persistence(task)
                for task_id, task in TASKS.items()
            }

        if not tasks_snapshot:
            return

        # Strip DataFrames from completed tasks (outside lock).
        # Leave DataFrames as-is for active tasks — save_task() handles
        # DataFrame serialization natively via pd.DataFrame.to_json().
        for task_copy in tasks_snapshot.values():
            task_copy.pop("persistence_failed", None)
            if task_copy.get("status") == "completed" and "dataframe" in task_copy:
                del task_copy["dataframe"]

        success_count = 0
        for task_id, task_data in tasks_snapshot.items():
            if storage_manager.save_task(task_id, task_data):
                success_count += 1
        logger.info(
            "Bulk save completed: %s/%s tasks", success_count, len(tasks_snapshot)
        )
    except Exception:
        logger.exception("Bulk task save failed")


def load_single_task(task_id):
    """Load a single task from database into memory.

    The loaded task will NOT have a DataFrame — only process_annotation()
    needs one, and it always has it in memory from the upload step.
    Tasks loaded from DB are for read-only display (status, results, downloads).
    """
    try:
        task_data = storage_manager.get_task(task_id)
        if task_data:
            with TASKS_LOCK:
                # A concurrent request may have loaded and advanced this task
                # while the database read was in flight. Never overwrite the
                # newer in-memory state with this older snapshot.
                TASKS.setdefault(task_id, task_data)
            return True
        return False
    except Exception as e:
        logger.exception("Failed to load task %s: %s", task_id, e)
        return False


def is_valid_task_id(task_id):
    """Return whether a value is a canonical UUID task identifier."""
    if not isinstance(task_id, str):
        return False
    try:
        return str(uuid.UUID(task_id)) == task_id
    except (ValueError, AttributeError):
        return False


def parse_task_ids(data, *, limit=100):
    """Validate a bounded, duplicate-free bulk task ID list."""
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    task_ids = data.get("task_ids")
    if not isinstance(task_ids, list):
        raise ValueError("task_ids must be an array")
    if not task_ids:
        raise ValueError("task_ids must not be empty")
    if len(task_ids) > limit:
        raise ValueError(f"No more than {limit} tasks may be changed at once")
    if any(not is_valid_task_id(task_id) for task_id in task_ids):
        raise ValueError("Every task ID must be a valid UUID")
    return list(dict.fromkeys(task_ids))


def ensure_task_loaded(task_id):
    """Ensure a task is available in TASKS, loading from DB if needed.

    Single point of truth for the "check memory → fall back to DB" pattern
    used by every task endpoint.  Returns True if task is in TASKS.
    """
    if not is_valid_task_id(task_id):
        return False
    with TASKS_LOCK:
        if task_id in TASKS:
            return True
    return load_single_task(task_id)


def ensure_tasks_loaded(task_ids):
    """Ensure multiple tasks are in TASKS, loading missing ones from DB.

    Bulk variant of ensure_task_loaded().  I/O happens outside the lock.
    """
    valid_ids = [task_id for task_id in task_ids if is_valid_task_id(task_id)]
    with TASKS_LOCK:
        missing = [tid for tid in valid_ids if tid not in TASKS]
    for tid in missing:
        load_single_task(tid)


def get_session_owner_id():
    """Return a stable anonymous owner ID stored in the signed session."""
    owner_id = session.get("owner_id")
    if not isinstance(owner_id, str) or len(owner_id) < 32:
        owner_id = secrets.token_urlsafe(24)
        session["owner_id"] = owner_id
    return owner_id


def check_task_ownership(task_id):
    """Verify session ownership, with an IP fallback for legacy tasks."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return False
        owner_id = task.get("owner_id")
        if owner_id:
            session_owner_id = session.get("owner_id", "")
            return isinstance(session_owner_id, str) and secrets.compare_digest(
                owner_id, session_owner_id
            )
        if task_id in session.get("owned_task_ids", []):
            return True
        if task.get("owner_session_required"):
            return False
        client_ip = get_client_ip()
        task_ip = task.get("ip_address")
        return bool(task_ip) and task_ip == client_ip


def get_all_tasks_merged():
    """Merge DB history with in-memory state for admin views.

    DB provides the full task history.  Memory overlays the freshest
    state for any task that exists in TASKS (progress, heartbeat, etc.).
    Falls back to memory-only if DB is unavailable.
    """
    try:
        merged = storage_manager.get_all_tasks()
    except Exception as e:
        logger.warning("Task history unavailable; using memory snapshot: %s", e)
        merged = {}
    with TASKS_LOCK:
        for task_id, task in TASKS.items():
            merged[task_id] = task
    return merged


def load_active_tasks():
    """Load active tasks (processing/queued) from database for monitoring.

    Merges DB state into memory without overwriting in-flight tasks.
    In-memory state is authoritative in this single-process model —
    DB state is a lightweight snapshot (no DataFrame/results).
    """
    try:
        active_tasks = storage_manager.load_active_tasks()

        # Strip dataframes (load_active_tasks returns lightweight rows,
        # but guard against future changes)
        for task_data in active_tasks.values():
            task_data.pop("dataframe", None)

        # Only add tasks NOT already in memory — never overwrite in-flight state
        with TASKS_LOCK:
            for task_id, task_data in active_tasks.items():
                if task_id not in TASKS:
                    TASKS[task_id] = task_data

            # Evict old terminal tasks from memory (>25 min after completion)
            current_time = utc_now()
            tasks_to_remove = []
            for task_id, task in TASKS.items():
                if task_id not in active_tasks:
                    status = task.get("status")
                    if status in TERMINAL_STATES:
                        completed_at = (
                            task.get("completed_at")
                            or task.get("failed_at")
                            or task.get("updated_at")
                        )
                        if completed_at:
                            try:
                                completed_time = parse_timestamp(completed_at)
                                if (
                                    current_time - completed_time
                                ).total_seconds() > 1500:
                                    tasks_to_remove.append(task_id)
                            except (ValueError, TypeError):
                                tasks_to_remove.append(task_id)
                        else:
                            tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del TASKS[task_id]

        for task_id in tasks_to_remove:
            logger.info("Removed old terminal task %s from memory", task_id)

        logger.info(
            "Loaded %s active tasks and removed %s old tasks",
            len(active_tasks),
            len(tasks_to_remove),
        )
        return True
    except Exception as e:
        logger.exception("Failed to load active tasks: %s", e)
        return False


def require_admin(f):
    """Decorator to require admin authentication"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not ADMIN_ENABLED:
            return jsonify({"error": "Admin is not configured"}), 404
        if not session.get("is_admin"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated_function


load_active_tasks()


def _bg_save(task_id, label, retries=0):
    """Persist task state from a background thread.

    Unlike request handlers (which return 500 on save failure), background
    threads have no HTTP response to send.  Failures are logged instead.
    Use retries > 0 for critical saves (e.g. completed results) where a
    transient failure should be retried before giving up.
    """
    for attempt in range(retries + 1):
        if save_single_task(task_id, label):
            return True
        if attempt < retries:
            logger.warning(
                "Failed to persist %s for task %s; retrying (%s/%s)",
                label,
                task_id,
                attempt + 1,
                retries,
            )
            time.sleep(1)
    log = logger.critical if retries > 0 else logger.warning
    log("Failed to persist %s for task %s", label, task_id)
    return False


def _start_annotation_worker(arguments):
    """Start one named annotation worker and return its thread handle."""
    thread = threading.Thread(
        target=process_annotation,
        args=arguments,
        name=f"annotation-{arguments[0]}",
        daemon=True,
    )
    thread.start()
    return thread


# Recover any stuck tasks from previous runs
def recover_stuck_tasks_on_startup():
    """Fail orphaned tasks because background threads cannot survive restart."""
    try:
        tasks_to_save = []

        with TASKS_LOCK:
            for task_id, task in TASKS.items():
                try:
                    state_machine = TaskStateMachine(task)
                    state = state_machine.get_state()
                    if state in {TaskState.PROCESSING, TaskState.QUEUED}:
                        state_machine.transition_to(
                            TaskState.FAILED,
                            "Task was interrupted by an application restart",
                        )
                        tasks_to_save.append(task_id)
                except TaskStateError as error:
                    logger.error("Cannot recover task %s: %s", task_id, error)

        # I/O outside lock
        for task_id in tasks_to_save:
            _bg_save(task_id, "startup_recovery")

        logger.info("Recovered %s orphaned startup tasks", len(tasks_to_save))

    except Exception as e:
        logger.exception("Startup task recovery failed: %s", e)


# Run recovery on startup
recover_stuck_tasks_on_startup()

# Start memory cleanup thread
memory_cleanup_thread = MemoryCleanupThread(interval=300)  # Clean every 5 minutes
memory_cleanup_thread.start()

# Start task monitoring thread
task_monitoring_thread = TaskMonitoringThread(interval=60)  # Check every minute
task_monitoring_thread.start()

# Cleanup is handled by a single atexit handler (see cleanup() near end of file)


def setup_mllmcelltype():
    """Load the annotation entry point."""
    try:
        from mllmcelltype import interactive_consensus_annotation

        logger.info("mLLMCelltype package loaded")
        return True, interactive_consensus_annotation
    except ImportError:
        logger.exception("mLLMCelltype package is unavailable")
        return False, None


MLLM_AVAILABLE, interactive_consensus_annotation = setup_mllmcelltype()


def create_app():
    """Create Flask application with enhanced security"""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
    # x_for=0: do NOT trust X-Forwarded-For for client IP (spoofable).
    # Client IP is obtained via X-Real-IP set by Caddy (see get_client_ip).
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=0, x_proto=1, x_host=1, x_prefix=1)
    secret_key = os.environ.get("FLASK_SECRET_KEY", "")
    if not secret_key:
        if not LOCAL_TESTING:
            raise RuntimeError("FLASK_SECRET_KEY is required")
        secret_key = secrets.token_hex(32)
    if not LOCAL_TESTING and len(secret_key) < 32:
        raise RuntimeError("FLASK_SECRET_KEY must contain at least 32 characters")
    app.secret_key = secret_key

    # Session cookie security.
    # Secure flag is scheme-aware via _SchemeAwareSessionInterface (HTTPS → Secure).
    app.config["SESSION_COOKIE_SECURE"] = False
    app.config["SESSION_COOKIE_HTTPONLY"] = True  # No JS access
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # CSRF protection
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)
    app.session_interface = _SchemeAwareSessionInterface()

    # Enhanced security headers and logging for SEO and Core Web Vitals
    @app.after_request
    def after_request_handler(response):
        """Add security headers"""
        if request.is_secure:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Browser code talks only to this application; provider calls are server-side.
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "font-src 'self'; "
            "img-src 'self' data: https: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        response.headers["Content-Security-Policy"] = csp_policy

        # Additional security headers for SEO benefits
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # Performance hints for Core Web Vitals
        if request.endpoint == "index":
            response.headers["Link"] = (
                "</static/css/style.css>; rel=preload; as=style, "
                "</static/js/app.js>; rel=preload; as=script, "
                "</static/js/vue.global.js>; rel=preload; as=script"
            )

        return response

    # Rate limiting is mandatory because several routes trigger paid API calls.
    try:
        from config.rate_limiter import create_limiter

        limiter = create_limiter(app)
        logger.info("Rate limiting enabled")
    except Exception as e:
        raise RuntimeError(
            f"Rate limiting is required but failed to initialize: {e}"
        ) from e

    # Configure Jinja2 to avoid conflicts with Vue.js
    # IMPORTANT: Use {[{ }]} instead of {{ }} in ALL template files
    # This prevents conflicts with Vue.js template syntax
    # DO NOT change this to {{ }} - it will break Vue.js functionality
    app.jinja_env.variable_start_string = "{[{"
    app.jinja_env.variable_end_string = "}]}"

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/faq")
    def faq():
        """FAQ page optimized for AI search engines"""
        return render_template("faq.html")

    @app.route("/scrna-troubleshooting-guide")
    def scrna_troubleshooting_guide():
        """Complete troubleshooting guide for scRNA-seq analysis problems"""
        return render_template("scrna_troubleshooting_guide.html")

    @app.route("/annotation-accuracy")
    @app.route("/best-annotation-tool")
    @app.route("/best-cell-annotation-tools")
    @app.route("/case-studies")
    @app.route("/cell-type-annotation")
    @app.route("/how-it-works")
    @app.route("/how-to-annotate")
    @app.route("/resources")
    @app.route("/scrna-annotation-guide")
    def retired_page():
        """Redirect retired content routes to the maintained homepage."""
        return redirect(url_for("index"), code=301)

    @app.route("/api/upload", methods=["POST"])
    @limiter.limit("30 per hour")
    def upload_file():
        """Handle file upload"""
        try:
            if "file" not in request.files:
                return jsonify({"error": "No file provided"}), 400

            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            # Generate task ID
            task_id = str(uuid.uuid4())

            try:
                file_size = get_upload_size(file)
                df = read_marker_dataframe(file)
            except MarkerFileError as error:
                return jsonify({"error": str(error)}), 400

            try:
                marker_genes = convert_dataframe_to_marker_genes(df)
            except ValueError as error:
                return jsonify({"error": str(error)}), 400
            if not marker_genes:
                return jsonify(
                    {"error": "No valid marker genes were found in the file."}
                ), 400

            # Get client IP and location
            client_ip = get_client_ip()

            # Store task data under lock
            task_data = {
                "status": "file_ready",
                "filename": file.filename,
                "file_size": file_size,
                "dataframe": df,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "preview": to_json_compatible(df.head(5).to_dict("records")),
                "created_at": utc_now_iso(),
                "ip_address": client_ip,
                "cell_count": len(marker_genes),
                "owner_id": get_session_owner_id(),
                "owner_session_required": True,
            }
            with TASKS_LOCK:
                TASKS[task_id] = task_data

            # Persist the new task outside the shared lock.
            if not save_single_task(task_id, "file_upload"):
                # Roll back: remove from memory since DB save failed
                with TASKS_LOCK:
                    TASKS.pop(task_id, None)
                return jsonify({"error": "Internal server error"}), 500

            return jsonify(
                {
                    "success": True,
                    "task_id": task_id,
                    "filename": file.filename,
                    "file_info": {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "preview": to_json_compatible(df.head(5).to_dict("records")),
                    },
                }
            )

        except Exception:
            logger.exception("File upload failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/test-api-key", methods=["POST"])
    @limiter.limit("30 per hour")
    def test_api_key():
        """Test API key validity for a specific provider"""
        try:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            provider = data.get("provider")
            api_key = data.get("api_key")
            model = data.get("model")

            if not isinstance(provider, str) or not isinstance(api_key, str):
                return jsonify({"error": "Provider and API key are required"}), 400
            provider = provider.strip().lower()
            api_key = api_key.strip()
            if provider not in SUPPORTED_PROVIDERS:
                return jsonify({"error": "Unsupported provider"}), 400
            if not api_key or len(api_key) > MAX_API_KEY_LENGTH:
                return jsonify({"error": "API key has an invalid length"}), 400
            if model is not None:
                if not isinstance(model, str) or not model.strip() or len(model) > 300:
                    return jsonify({"error": "Model must be a non-empty string"}), 400
                model = model.strip()

            # Import lightweight validator (Web-specific utility)
            from utils.api_validator import test_provider_api

            # Execute API test
            result = test_provider_api(provider, api_key, model)

            if result["valid"]:
                return jsonify(
                    {
                        "valid": True,
                        "message": result["message"],
                        "response_time": result.get("response_time", 0),
                    }
                )
            return jsonify({"valid": False, "error": result["error"]}), 400

        except Exception:
            logger.exception("API key test failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/tasks/<task_id>/reset", methods=["POST"])
    def reset_task(task_id):
        """Reset a completed/failed/timeout task back to file_ready for rerun.

        This is the user-facing counterpart of the admin retry endpoint.
        It allows users to adjust parameters and rerun their own task
        without uploading the file again. Only the owning signed session may
        reset the task.
        """
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            if not check_task_ownership(task_id):
                return jsonify({"error": "Permission denied"}), 403

            with TASKS_LOCK:
                task = TASKS[task_id]
                current_status = task.get("status", "")
                if current_status not in TERMINAL_STATES:
                    return jsonify(
                        {"error": f"Task cannot be reset from state: {current_status}"}
                    ), 400

                snapshot = _snapshot_task(task)
                state_machine = TaskStateMachine(task)
                state_machine.transition_to(TaskState.FILE_READY)
                reset_version = task["state_version"]

            if not save_single_task(task_id, "user_reset"):
                _restore_task_snapshot_if_unchanged(
                    task_id,
                    snapshot,
                    expected_status=TaskState.FILE_READY.value,
                    expected_version=reset_version,
                )
                return jsonify({"error": "Internal server error"}), 500

            return jsonify({"success": True, "message": "Task reset to file_ready"})
        except Exception:
            logger.exception("Failed to reset task %s", task_id)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/annotate", methods=["POST"])
    @limiter.limit("20 per hour")
    def start_annotation():
        """Start annotation process"""
        try:
            if not MLLM_AVAILABLE or not callable(interactive_consensus_annotation):
                return jsonify({"error": "Annotation service is unavailable"}), 503

            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            try:
                annotation_request = parse_annotation_request(data)
            except AnnotationValidationError as error:
                return jsonify({"error": str(error)}), 400

            task_id = annotation_request.task_id

            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            if not check_task_ownership(task_id):
                return jsonify({"error": "Permission denied"}), 403

            logger.info("Annotation requested for task %s", task_id)

            # Ensure DataFrame is available (may be absent after server restart)
            with TASKS_LOCK:
                has_dataframe = "dataframe" in TASKS[task_id]

            if not has_dataframe:
                # Restore from database (I/O outside lock)
                task_from_db = storage_manager.get_task(task_id, include_dataframe=True)
                if task_from_db and "dataframe" in task_from_db:
                    with TASKS_LOCK:
                        TASKS[task_id]["dataframe"] = task_from_db["dataframe"]
                    logger.info("Restored input data for task %s", task_id)
                else:
                    return jsonify(
                        {
                            "error": "Original file data is no longer available. Please upload the file again."
                        }
                    ), 410

            species = annotation_request.species
            tissue = annotation_request.tissue
            models = annotation_request.models
            api_keys = annotation_request.api_keys
            consensus_threshold = annotation_request.consensus_threshold
            entropy_threshold = annotation_request.entropy_threshold
            max_rounds = annotation_request.max_rounds
            consensus_model = annotation_request.consensus_model

            with TASKS_LOCK:
                task = TASKS[task_id]
                current_status = task.get("status", "")
                if current_status != "file_ready":
                    return jsonify(
                        {
                            "error": f"Task cannot be started from state: {current_status}"
                        }
                    ), 400

            # Enforce concurrency limit after all validation passes.
            # Acquiring here (rather than earlier) avoids leaking a slot
            # when a pre-flight check returns an error.
            if not _annotation_slots.acquire(blocking=False):
                return jsonify(
                    {"error": "Server is at capacity. Please try again later."}
                ), 429

            run_id = str(uuid.uuid4())

            # From this point, every exit path must release the slot — either
            # via process_annotation's finally block (thread started) or
            # explicitly below (any failure before thread.start).
            try:
                with TASKS_LOCK:
                    task = TASKS[task_id]
                    current_status = task.get("status", "")
                    if current_status != TaskState.FILE_READY.value:
                        _annotation_slots.release()
                        return jsonify(
                            {
                                "error": (
                                    "Task state changed before it could be started: "
                                    f"{current_status}"
                                )
                            }
                        ), 409
                    snapshot = _snapshot_task(task)
                    task["species"] = species
                    task["tissue"] = tissue
                    task["models"] = list(models)
                    task["consensus_threshold"] = consensus_threshold
                    task["entropy_threshold"] = entropy_threshold
                    task["max_rounds"] = max_rounds
                    task["consensus_model"] = consensus_model
                    state_machine = TaskStateMachine(task)
                    state_machine.transition_to(TaskState.QUEUED)
                    task["run_id"] = run_id
                    queued_version = task["state_version"]

                if not save_single_task(task_id, "queued"):
                    _restore_task_snapshot_if_unchanged(
                        task_id,
                        snapshot,
                        expected_status=TaskState.QUEUED.value,
                        expected_version=queued_version,
                    )
                    _annotation_slots.release()
                    return jsonify({"error": "Internal server error"}), 500

                # The worker will transition QUEUED -> PROCESSING.
                worker_arguments = (
                    task_id,
                    run_id,
                    species,
                    tissue,
                    models,
                    api_keys,
                    consensus_threshold,
                    entropy_threshold,
                    max_rounds,
                    consensus_model,
                )
                thread = _start_annotation_worker(worker_arguments)
            except Exception:
                _annotation_slots.release()
                _fail_annotation_task(
                    task_id,
                    "Failed to start processing thread",
                    expected_run_id=run_id,
                )
                return jsonify({"error": "Internal server error"}), 500

            with TASKS_LOCK:
                if (
                    task_id in TASKS
                    and TASKS[task_id].get("run_id") == run_id
                    and TASKS[task_id].get("status") in RUNNING_STATES
                ):
                    TASKS[task_id]["worker_thread_id"] = thread.ident

            return jsonify(
                {"success": True, "task_id": task_id, "message": "Annotation started"}
            )

        except Exception:
            logger.exception("Failed to start annotation")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/tasks/<task_id>")
    def get_task_status(task_id):
        """Get task status"""
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            if not check_task_ownership(task_id):
                return jsonify({"error": "Permission denied"}), 403

            # Copy all needed fields under lock to avoid reading mutable references outside lock
            with TASKS_LOCK:
                task = TASKS[task_id]
                status = task["status"]
                progress_value = copy.deepcopy(task.get("progress"))
                progress_copy = (
                    progress_value
                    if isinstance(progress_value, dict)
                    else {"percentage": progress_value}
                    if progress_value is not None
                    else None
                )
                error_msg = task.get("error")
                started_at = task.get("started_at")
                persistence_failed = task.get("persistence_failed", False)

            # Only log non-processing status checks to avoid flooding
            # (processing tasks are polled every 2s by the frontend)
            if status != "processing":
                logger.debug("Task %s status: %s", task_id, status)

            response = {"task_id": task_id, "status": status}

            if status == "file_ready":
                response["progress"] = 10
                response["message"] = "File processed, ready for annotation"
            elif status == "processing":
                # Compute progress: use real percentage if set, otherwise
                # estimate from elapsed time (grows 10→85% over ~10 min).
                # The mLLMCelltype package runs as one blocking call, so
                # per-cluster progress is unavailable.
                if progress_copy and progress_copy.get("percentage") is not None:
                    pct = min(
                        100.0, max(0.0, _finite_float(progress_copy["percentage"], 15))
                    )
                else:
                    pct = 15  # default: just started
                    if started_at:
                        try:
                            elapsed = max(
                                0,
                                (
                                    utc_now() - parse_timestamp(started_at)
                                ).total_seconds(),
                            )
                            pct = min(85, 10 + int(elapsed / 600 * 75))
                        except (ValueError, TypeError):
                            pass

                stage = "Processing"
                msg = "Processing annotations..."
                if progress_copy:
                    stage = str(
                        progress_copy.get("stage", "Processing") or "Processing"
                    )
                    msg = str(progress_copy.get("message", msg) or msg)

                response["progress"] = pct
                response["message"] = msg
                response["progress_details"] = {
                    "stage": stage,
                    "total_clusters": progress_copy.get("total", 0)
                    if progress_copy
                    else 0,
                    "phase": progress_copy.get("phase", "processing")
                    if progress_copy
                    else "processing",
                }
            elif status == "completed":
                response["progress"] = 100
                response["message"] = "Analysis completed successfully"
                if persistence_failed:
                    response["persistence_failed"] = True
            elif status == "failed":
                response["progress"] = 0
                response["message"] = error_msg or "Processing failed"
                response["error"] = error_msg or "Processing failed"
            elif status == "cancelled":
                response["progress"] = 0
                response["message"] = error_msg or "Task was cancelled"
                response["error"] = error_msg or "Task was cancelled"
            elif status == "queued":
                response["progress"] = 5
                response["message"] = "Task queued, waiting to start..."
            elif status == "timeout":
                response["progress"] = 0
                response["message"] = error_msg or "Task timed out"
                response["error"] = error_msg or "Task timed out"

            return jsonify(response)

        except Exception:
            logger.exception("Failed to read task status for %s", task_id)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/results/<task_id>")
    def get_results(task_id):
        """Get annotation results"""
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            if not check_task_ownership(task_id):
                return jsonify({"error": "Permission denied"}), 403

            # Only extract the lightweight fields the frontend actually uses.
            # Heavy fields (discussion_logs, model_annotations, resolved,
            # processing_details) stay in memory for download endpoints.
            with TASKS_LOCK:
                task = TASKS[task_id]
                if task["status"] != "completed":
                    return jsonify({"error": "Task not completed"}), 409
                results = task.get("results")
                if not results:
                    return jsonify(
                        {"error": "Results data is not available for this task"}
                    ), 404
                web_results = {
                    "consensus": copy.deepcopy(results.get("consensus", {})),
                    "consensus_proportion": copy.deepcopy(
                        results.get("consensus_proportion", {})
                    ),
                    "entropy": copy.deepcopy(results.get("entropy", {})),
                    "controversial_clusters": copy.deepcopy(
                        results.get("controversial_clusters", [])
                    ),
                    "metadata": copy.deepcopy(results.get("metadata", {})),
                }

            return jsonify({"task_id": task_id, **web_results})

        except Exception:
            logger.exception("Failed to read results for task %s", task_id)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/download/<task_id>/<format>")
    def download_results(task_id, format):
        """Download results in specified format"""
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            if not session.get("is_admin") and not check_task_ownership(task_id):
                return jsonify({"error": "Permission denied"}), 403

            with TASKS_LOCK:
                task = TASKS[task_id]
                if task["status"] != "completed":
                    return jsonify({"error": "Results not ready"}), 409

                results = task.get("results")
                if not results:
                    return jsonify(
                        {"error": "Results data is not available for this task"}
                    ), 404

                results = copy.deepcopy(results)

            df = create_results_dataframe(results)

            # Write to in-memory buffer (no temp file leak)
            buf = io.BytesIO()

            if format == "csv":
                buf.write(df.to_csv(index=False).encode("utf-8"))
                download_name = f"annotation_results_{task_id}.csv"
                mimetype = "text/csv"
            elif format == "tsv":
                buf.write(df.to_csv(index=False, sep="\t").encode("utf-8"))
                download_name = f"annotation_results_{task_id}.tsv"
                mimetype = "text/tab-separated-values"
            elif format in ["excel", "xlsx"]:
                df.to_excel(buf, index=False, engine="openpyxl")
                download_name = f"annotation_results_{task_id}.xlsx"
                mimetype = (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                return jsonify({"error": "Unsupported format"}), 400

            buf.seek(0)
            return send_file(
                buf, as_attachment=True, download_name=download_name, mimetype=mimetype
            )

        except Exception:
            logger.exception("Failed to download task %s as %s", task_id, format)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/download-logs/<task_id>")
    def download_annotation_logs(task_id):
        """Download annotation logs and discussion details"""
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            if not session.get("is_admin") and not check_task_ownership(task_id):
                return jsonify({"error": "Permission denied"}), 403

            with TASKS_LOCK:
                task = TASKS[task_id]
                if task["status"] != "completed":
                    return jsonify({"error": "Logs not ready"}), 409
                results = copy.deepcopy(task.get("results", {}))
                task_snapshot = {
                    "id": task_id,
                    "created_at": task.get("created_at", ""),
                    "species": task.get("species", ""),
                    "tissue": task.get("tissue", ""),
                    "models": task.get("models", []).copy(),
                }

            # Create comprehensive log content
            log_content = create_annotation_log_content(results, task_snapshot)

            buf = io.BytesIO(log_content.encode("utf-8"))
            return send_file(
                buf,
                as_attachment=True,
                download_name=f"annotation_logs_{task_id}.txt",
                mimetype="text/plain",
            )

        except Exception:
            logger.exception("Failed to download logs for task %s", task_id)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/download-discussion/<task_id>")
    def download_discussion_details(task_id):
        """Download detailed discussion logs in JSON format"""
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            if not session.get("is_admin") and not check_task_ownership(task_id):
                return jsonify({"error": "Permission denied"}), 403

            with TASKS_LOCK:
                task = TASKS[task_id]
                if task["status"] != "completed":
                    return jsonify({"error": "Discussion logs not ready"}), 409
                results = copy.deepcopy(task.get("results", {}))
                task_info = {
                    "task_id": task_id,
                    "timestamp": task.get("created_at", ""),
                    "species": task.get("species", ""),
                    "tissue": task.get("tissue", ""),
                    "models": task.get("models", []).copy(),
                }

            discussion_data = {
                "task_info": task_info,
                "discussion_logs": results.get("discussion_logs", {}),
                "model_annotations": results.get("model_annotations", {}),
                "resolved": results.get("resolved", {}),
                "controversial_clusters": results.get("controversial_clusters", []),
                "metadata": results.get("metadata", {}),
            }

            buf = io.BytesIO(
                json.dumps(discussion_data, indent=2, ensure_ascii=False).encode(
                    "utf-8"
                )
            )
            return send_file(
                buf,
                as_attachment=True,
                download_name=f"discussion_details_{task_id}.json",
                mimetype="application/json",
            )

        except Exception:
            logger.exception("Failed to download discussion for task %s", task_id)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/sample")
    def get_sample_data():
        """Download sample data file"""
        try:
            sample_path = Path(__file__).resolve().parent / "data" / "sample_data.csv"
            if sample_path.is_file():
                return send_file(
                    sample_path, as_attachment=True, download_name="sample_data.csv"
                )
            return jsonify({"error": "Sample file not found"}), 404
        except Exception:
            logger.exception("Failed to download sample data")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/health")
    def health_check():
        if not MLLM_AVAILABLE:
            return jsonify(
                {
                    "status": "unhealthy",
                    "annotation_engine": "unavailable",
                }
            ), 503
        return jsonify(
            {
                "status": "healthy",
                "annotation_engine": "available",
            }
        )

    @app.route("/api/deployment-info")
    def deployment_info():
        """Get deployment information"""
        deploy_time = os.environ.get("DEPLOY_TIME", "")

        # Validate ISO-8601 shape; fall back to process start time
        # so the footer shows a stable value instead of changing on
        # every page refresh.
        if not deploy_time or "-" not in deploy_time or "T" not in deploy_time:
            deploy_time = _APP_START_TIME

        return jsonify(
            {
                "annotation_engine_version": _ANNOTATION_ENGINE_VERSION,
                "deploy_time": deploy_time,
                "version": os.environ.get("VERSION", "latest"),
            }
        )

    @app.route("/api/provider-catalog")
    @app.route("/api/provider-defaults")
    def provider_catalog():
        """Expose the curated provider model catalog to the UI."""
        providers = sorted(SUPPORTED_PROVIDERS.intersection(MODEL_CATALOG))
        serialized_catalog = get_serialized_catalog()
        defaults = get_provider_defaults()
        provider_names = get_provider_names()
        return jsonify(
            {
                "providers": providers,
                "provider_names": {
                    provider: provider_names[provider] for provider in providers
                },
                "defaults": {provider: defaults[provider] for provider in providers},
                "models": {
                    provider: serialized_catalog[provider] for provider in providers
                },
                "updated_at": MODEL_CATALOG_UPDATED_AT,
            }
        )

    @app.route("/admin/login")
    def admin_login_page():
        """Admin login page"""
        if not ADMIN_ENABLED:
            return "Admin dashboard is not configured.", 404
        return render_template("admin-login.html")

    @app.route("/admin/login", methods=["POST"])
    @limiter.limit("10 per minute")
    def admin_login():
        """Admin login endpoint"""
        if not ADMIN_ENABLED:
            return jsonify({"error": "Admin is not configured"}), 404
        try:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400
            username = data.get("username")
            password = data.get("password")

            if not username or not password:
                return jsonify({"error": "Invalid credentials"}), 401

            valid_username = secrets.compare_digest(str(username), ADMIN_USERNAME)
            valid_password = check_password_hash(ADMIN_PASSWORD_HASH, str(password))
            if valid_username and valid_password:
                session.permanent = True
                session["is_admin"] = True
                session["admin_login_time"] = utc_now_iso()
                return jsonify({"success": True})
            return jsonify({"error": "Invalid credentials"}), 401
        except Exception:
            logger.exception("Admin login failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/admin/logout", methods=["POST"])
    def admin_logout():
        """Admin logout endpoint"""
        session.pop("is_admin", None)
        session.pop("admin_login_time", None)
        session.permanent = False
        return jsonify({"success": True})

    @app.route("/admin")
    @require_admin
    def admin_dashboard():
        """Admin dashboard page"""
        return render_template("admin.html")

    @app.route("/api/admin/stats")
    @require_admin
    def get_admin_dashboard_stats():
        """Return dashboard metrics derived from persisted task facts."""
        try:
            today = utc_now().date().isoformat()
            merged = get_all_tasks_merged()

            user_first_seen = {}
            active_user_ids = set()
            durations = []
            completed_count = 0
            failed_count = 0
            processing_count = 0
            waiting_count = 0
            completed_today = 0
            last_error_at = None

            for task in merged.values():
                status = task.get("status")
                ip_address = task.get("ip_address")
                user_id = task.get("owner_id") or ip_address
                created_at = str(task.get("created_at") or "")
                if user_id and user_id != "unknown":
                    previous = user_first_seen.get(user_id)
                    if not previous or created_at < previous:
                        user_first_seen[user_id] = created_at
                    if status in RUNNING_STATES:
                        active_user_ids.add(user_id)

                if status == "processing":
                    processing_count += 1
                elif status in {"file_ready", "queued"}:
                    waiting_count += 1
                elif status == "completed":
                    completed_count += 1
                    completed_at = str(task.get("completed_at") or "")
                    if completed_at.startswith(today):
                        completed_today += 1
                    if task.get("started_at") and task.get("completed_at"):
                        try:
                            duration = (
                                parse_timestamp(task["completed_at"])
                                - parse_timestamp(task["started_at"])
                            ).total_seconds()
                            if duration >= 0:
                                durations.append(duration)
                        except (TypeError, ValueError):
                            pass
                elif status in {"failed", "timeout"}:
                    failed_count += 1
                    error_at = str(
                        task.get("failed_at") or task.get("updated_at") or ""
                    )
                    if error_at and (last_error_at is None or error_at > last_error_at):
                        last_error_at = error_at

            finished_count = completed_count + failed_count
            success_rate = (
                round(completed_count / finished_count * 100, 1)
                if finished_count
                else 0.0
            )
            error_rate = (
                round(failed_count / finished_count * 100, 1) if finished_count else 0.0
            )

            sorted_tasks = sorted(
                merged.items(),
                key=lambda item: str(item[1].get("created_at") or ""),
                reverse=True,
            )[:10]

            recent_tasks = []
            for task_id, task in sorted_tasks:
                progress = task.get("progress", {})
                recent_tasks.append(
                    {
                        "id": task_id,
                        "status": task.get("status", "unknown"),
                        "created_at": task.get("created_at", ""),
                        "filename": task.get("filename", ""),
                        "ip_address": task.get("ip_address", "unknown"),
                        "models": task.get("models", []),
                        "progress": (
                            _finite_float(progress.get("percentage", 0))
                            if isinstance(progress, dict)
                            else _finite_float(progress)
                        ),
                    }
                )

            return jsonify(
                {
                    "stats": {
                        "total_users": len(user_first_seen),
                        "new_users_today": sum(
                            1
                            for first_seen in user_first_seen.values()
                            if first_seen.startswith(today)
                        ),
                        "active_users": len(active_user_ids),
                        "active_tasks": processing_count + waiting_count,
                        "processing_tasks": processing_count,
                        "waiting_tasks": waiting_count,
                        "completed_today": completed_today,
                        "success_rate": success_rate,
                        "avg_time_seconds": (
                            round(sum(durations) / len(durations))
                            if durations
                            else None
                        ),
                        "failed_tasks": failed_count,
                        "error_rate": error_rate,
                        "last_error_at": last_error_at,
                    },
                    "recent_tasks": recent_tasks,
                }
            )
        except Exception:
            logger.exception("Failed to build admin statistics")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/admin/tasks/<task_id>/cancel", methods=["POST"])
    @require_admin
    def cancel_task(task_id):
        """Cancel a running task"""
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            with TASKS_LOCK:
                task = TASKS[task_id]
                current_status = task.get("status", "")

                if current_status not in CANCELLABLE_STATES:
                    return jsonify(
                        {"error": f"Cannot cancel task with status: {current_status}"}
                    ), 400

                # Update task status via state machine
                state_machine = TaskStateMachine(task)
                state_machine.transition_to(
                    TaskState.CANCELLED, "Task cancelled by administrator"
                )
                task["cancelled_by"] = "admin"

            # I/O outside lock
            if not _bg_save(task_id, "admin_cancel", retries=2):
                with TASKS_LOCK:
                    if task_id in TASKS:
                        TASKS[task_id]["persistence_failed"] = True
                _invalidate_admin_cache()
                return jsonify(
                    {
                        "success": True,
                        "task_id": task_id,
                        "message": "Task cancelled in memory, but persistence failed",
                        "previous_status": current_status,
                        "persistence_failed": True,
                    }
                ), 202
            _invalidate_admin_cache()

            return jsonify(
                {
                    "success": True,
                    "task_id": task_id,
                    "message": f"Task {task_id} has been cancelled",
                    "previous_status": current_status,
                }
            )

        except Exception:
            logger.exception("Failed to cancel task %s", task_id)
            return jsonify({"error": "Internal server error"}), 500

    # Simple cache for admin tasks
    _admin_tasks_cache = {
        "data": None,
        "timestamp": None,
        "key": None,
        "ttl": 5,  # 5 seconds cache
    }

    def _snapshot_task(task):
        """Shallow-copy a task dict with an isolated state_history list.

        transition_to() appends to state_history in place, so a plain
        dict(task) would leave the snapshot and the live task sharing the
        same list — polluting the snapshot on rollback.  Copying the list
        (but not its immutable-dict entries) is sufficient because
        existing entries are never mutated.
        """
        snap = dict(task)
        if "state_history" in snap:
            snap["state_history"] = list(snap["state_history"])
        return snap

    def _restore_task_snapshot_if_unchanged(
        task_id, snapshot, *, expected_status, expected_version
    ):
        """Rollback only when no concurrent transition has superseded a change."""
        with TASKS_LOCK:
            current = TASKS.get(task_id)
            if current is None:
                return False
            if (
                current.get("status") != expected_status
                or current.get("state_version", 0) != expected_version
            ):
                logger.warning(
                    "Skipped stale rollback for task %s (status=%s, version=%s)",
                    task_id,
                    current.get("status"),
                    current.get("state_version", 0),
                )
                return False
            TASKS[task_id] = snapshot
            return True

    def _invalidate_admin_cache():
        """Invalidate admin tasks cache after mutation operations."""
        _admin_tasks_cache["timestamp"] = None

    @app.route("/api/admin/all-tasks")
    @require_admin
    def get_all_tasks():
        """Get all tasks for admin management"""
        try:
            # Get pagination parameters
            page = max(1, request.args.get("page", 1, type=int) or 1)
            requested_per_page = request.args.get("per_page", 50, type=int)
            per_page = (
                0
                if requested_per_page == 0
                else max(1, min(requested_per_page or 50, 200))
            )
            use_cache = request.args.get("cache", "true").lower() == "true"

            # Check cache first (only if not paginated or same page)
            cache_key = f"{page}_{per_page}"
            now = utc_now()
            if (
                use_cache
                and _admin_tasks_cache["data"] is not None
                and _admin_tasks_cache["timestamp"] is not None
                and _admin_tasks_cache.get("key") == cache_key
                and (now - _admin_tasks_cache["timestamp"]).total_seconds()
                < _admin_tasks_cache["ttl"]
            ):
                logger.debug("Returning cached admin task page %s", page)
                return jsonify(_admin_tasks_cache["data"])
            # Merge DB history with in-memory state so admin sees all tasks
            merged = get_all_tasks_merged()

            all_tasks = []
            for task_id, task in merged.items():
                task_copy = {
                    "id": task_id,
                    "status": task.get("status", "unknown"),
                    "created_at": task.get("created_at", ""),
                    "filename": task.get("filename", ""),
                    "completed_at": task.get("completed_at", ""),
                    "cancelled_at": task.get("cancelled_at", ""),
                    "error": task.get("error", ""),
                    "ip_address": task.get("ip_address", "unknown"),
                    "models": task.get("models", []),
                    "started_at": task.get("started_at", ""),
                }

                if (
                    task.get("status") == "completed"
                    and task.get("started_at")
                    and task.get("completed_at")
                ):
                    try:
                        start = parse_timestamp(task["started_at"])
                        end = parse_timestamp(task["completed_at"])
                        task_copy["duration"] = (end - start).total_seconds()
                    except (ValueError, TypeError):
                        pass

                task_progress = task.get("progress", {})
                if isinstance(task_progress, dict):
                    task_copy["progress"] = _finite_float(
                        task_progress.get("percentage"),
                        100 if task.get("status") == "completed" else 0,
                    )
                else:
                    task_copy["progress"] = _finite_float(
                        task_progress,
                        100 if task.get("status") == "completed" else 0,
                    )

                all_tasks.append(task_copy)

            # Sort by created_at descending
            all_tasks.sort(
                key=lambda task: str(task.get("created_at") or ""), reverse=True
            )

            # Apply pagination (per_page=0 means return all)
            total_tasks = len(all_tasks)
            if per_page > 0:
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                paginated_tasks = all_tasks[start_idx:end_idx]
                total_pages = (total_tasks + per_page - 1) // per_page
            else:
                paginated_tasks = all_tasks
                total_pages = 1

            # Prepare response
            response_data = {
                "tasks": paginated_tasks,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total_tasks,
                    "pages": total_pages,
                },
            }

            # Update cache
            _admin_tasks_cache["data"] = response_data
            _admin_tasks_cache["timestamp"] = utc_now()
            _admin_tasks_cache["key"] = cache_key

            return jsonify(response_data)
        except Exception:
            logger.exception("Failed to load admin task list")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/admin/task/<task_id>")
    @require_admin
    def get_admin_task_details(task_id):
        """Get detailed information about a specific task"""
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            with TASKS_LOCK:
                original_task = TASKS[task_id]
                task = {}

                safe_fields = [
                    "status",
                    "filename",
                    "file_size",
                    "created_at",
                    "started_at",
                    "completed_at",
                    "cancelled_at",
                    "error",
                    "models",
                    "ip_address",
                    "species",
                    "tissue",
                ]

                for field in safe_fields:
                    if field in original_task:
                        value = original_task[field]
                        if isinstance(value, (str, int, float, bool, type(None))):
                            task[field] = value
                        elif isinstance(value, (list, dict)):
                            try:
                                cleaned_value = to_json_compatible(value)
                                json.dumps(cleaned_value)
                                task[field] = cleaned_value
                            except (TypeError, ValueError):
                                task[field] = str(value)
                        else:
                            task[field] = str(value)

                task["id"] = task_id

                try:
                    progress_data = original_task.get("progress", {})
                    if isinstance(progress_data, dict):
                        task["progress"] = _finite_float(
                            progress_data.get("percentage", 0)
                        )
                        task["progress_message"] = str(progress_data.get("message", ""))
                    elif progress_data is not None:
                        task["progress"] = _finite_float(progress_data)
                        task["progress_message"] = ""
                    else:
                        task["progress"] = (
                            100 if task.get("status") == "completed" else 0
                        )
                        task["progress_message"] = ""
                except Exception as error:
                    logger.warning("Failed to read progress for %s: %s", task_id, error)
                    task["progress"] = 0
                    task["progress_message"] = ""

                task["logs"] = _state_history_logs(
                    copy.deepcopy(original_task.get("state_history", []))
                )

            # Add duration calculation
            try:
                if task.get("started_at") and task.get("completed_at"):
                    started = parse_timestamp(task["started_at"])
                    completed = parse_timestamp(task["completed_at"])
                    duration_seconds = (completed - started).total_seconds()
                    task["duration"] = int(duration_seconds)
                else:
                    task["duration"] = None
            except Exception as error:
                logger.warning(
                    "Failed to calculate duration for %s: %s", task_id, error
                )
                task["duration"] = None

            return jsonify(task)

        except Exception:
            logger.exception("Failed to load details for task %s", task_id)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/admin/bulk-cancel", methods=["POST"])
    @require_admin
    def bulk_cancel_tasks():
        """Cancel multiple tasks"""
        try:
            data = request.get_json(silent=True)
            try:
                task_ids = parse_task_ids(data)
            except ValueError as error:
                return jsonify({"error": str(error)}), 400

            ensure_tasks_loaded(task_ids)

            cancelled = []
            failed = []
            with TASKS_LOCK:
                for task_id in task_ids:
                    if (
                        task_id in TASKS
                        and TASKS[task_id]["status"] in CANCELLABLE_STATES
                    ):
                        try:
                            state_machine = TaskStateMachine(TASKS[task_id])
                            state_machine.transition_to(
                                TaskState.CANCELLED,
                                "Task cancelled by administrator",
                            )
                            TASKS[task_id]["cancelled_by"] = "admin"
                            cancelled.append(task_id)
                        except Exception as error:
                            logger.warning(
                                "Failed to cancel task %s: %s", task_id, error
                            )
                            failed.append(task_id)
                    else:
                        failed.append(task_id)

            # Save outside lock — track persistence failures
            save_failed = []
            for task_id in cancelled:
                if not save_single_task(task_id, "bulk_cancel"):
                    save_failed.append(task_id)

            # A running worker may already observe cancellation. Never resurrect
            # it on a persistence error; retain the terminal in-memory state.
            if save_failed:
                with TASKS_LOCK:
                    for tid in save_failed:
                        if tid in TASKS:
                            TASKS[tid]["persistence_failed"] = True

            if cancelled:
                _invalidate_admin_cache()

            result = {
                "cancelled": cancelled,
                "failed": failed,
                "message": f"Cancelled {len(cancelled)} tasks",
            }
            if save_failed:
                result["save_failed"] = save_failed
                result["message"] += f" ({len(save_failed)} failed to persist)"
                return jsonify(result), 207
            return jsonify(result)
        except Exception:
            logger.exception("Bulk task cancellation failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/admin/bulk-retry", methods=["POST"])
    @require_admin
    def bulk_retry_tasks():
        """Retry multiple failed tasks"""
        try:
            data = request.get_json(silent=True)
            try:
                task_ids = parse_task_ids(data)
            except ValueError as error:
                return jsonify({"error": str(error)}), 400

            ensure_tasks_loaded(task_ids)

            retried = []
            failed = []
            snapshots = {}
            retry_versions = {}

            # Retry transitions task back to FILE_READY so the user can
            # re-submit with API keys (which are not stored for security).
            with TASKS_LOCK:
                for task_id in task_ids:
                    if task_id in TASKS and TASKS[task_id]["status"] in (
                        "failed",
                        "timeout",
                    ):
                        try:
                            snapshots[task_id] = _snapshot_task(TASKS[task_id])
                            state_machine = TaskStateMachine(TASKS[task_id])
                            state_machine.transition_to(TaskState.FILE_READY)
                            TASKS[task_id]["retry_count"] = (
                                TASKS[task_id].get("retry_count", 0) + 1
                            )
                            TASKS[task_id]["retried_at"] = utc_now_iso()
                            retry_versions[task_id] = TASKS[task_id]["state_version"]
                            retried.append(task_id)
                        except Exception as error:
                            logger.warning(
                                "Failed to retry task %s: %s", task_id, error
                            )
                            failed.append(task_id)
                    else:
                        failed.append(task_id)

            # Save outside lock — track persistence failures
            save_failed = []
            for task_id in retried:
                if not save_single_task(task_id, "bulk_retry"):
                    save_failed.append(task_id)

            # Rollback save-failed tasks and remove from success list
            if save_failed:
                for tid in save_failed:
                    _restore_task_snapshot_if_unchanged(
                        tid,
                        snapshots[tid],
                        expected_status=TaskState.FILE_READY.value,
                        expected_version=retry_versions[tid],
                    )
                retried = [tid for tid in retried if tid not in save_failed]

            if retried:
                _invalidate_admin_cache()

            result = {
                "retried": retried,
                "failed": failed,
                "message": f"Retried {len(retried)} tasks",
            }
            if save_failed:
                result["save_failed"] = save_failed
                result["message"] += f" ({len(save_failed)} failed to persist)"
                return jsonify(result), 207
            return jsonify(result)
        except Exception:
            logger.exception("Bulk task retry failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/admin/bulk-delete", methods=["POST"])
    @require_admin
    def bulk_delete_tasks():
        """Delete multiple tasks"""
        try:
            data = request.get_json(silent=True)
            try:
                task_ids = parse_task_ids(data)
            except ValueError as error:
                return jsonify({"error": str(error)}), 400

            ensure_tasks_loaded(task_ids)

            # Refuse to delete tasks with running threads — cancel first
            skipped_active = []
            deletable = []
            with TASKS_LOCK:
                for task_id in task_ids:
                    if task_id in TASKS and TASKS[task_id].get("status") in {
                        TaskState.QUEUED.value,
                        TaskState.PROCESSING.value,
                    }:
                        skipped_active.append(task_id)
                    else:
                        deletable.append(task_id)

            # Delete from database (I/O outside lock)
            deleted = []
            for task_id in deletable:
                if storage_manager.delete_task(task_id):
                    deleted.append(task_id)
            failed = [task_id for task_id in deletable if task_id not in deleted]

            # Only remove from memory after successful DB deletion
            with TASKS_LOCK:
                for task_id in deleted:
                    TASKS.pop(task_id, None)
            if deleted:
                _invalidate_admin_cache()

            result = {
                "deleted": deleted,
                "failed": failed,
                "message": f"Deleted {len(deleted)} tasks",
            }
            if skipped_active:
                result["skipped_active"] = skipped_active
                result["message"] += (
                    f" ({len(skipped_active)} active tasks skipped, cancel first)"
                )
            if failed:
                result["message"] += f" ({len(failed)} tasks could not be deleted)"
            if skipped_active or failed:
                return jsonify(result), 207
            return jsonify(result)
        except Exception:
            logger.exception("Bulk task deletion failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/admin/retry-task/<task_id>", methods=["POST"])
    @require_admin
    def retry_single_task(task_id):
        """Retry a single failed task"""
        try:
            if not ensure_task_loaded(task_id):
                return jsonify({"error": "Task not found"}), 404

            with TASKS_LOCK:
                task = TASKS[task_id]
                if task["status"] not in ("failed", "timeout"):
                    return jsonify(
                        {"error": "Task is not in failed or timeout state"}
                    ), 400

                snapshot = _snapshot_task(task)
                state_machine = TaskStateMachine(task)
                # Move back to FILE_READY so the user can re-submit with API keys
                # (API keys are not stored for security).
                state_machine.transition_to(TaskState.FILE_READY)
                task["retry_count"] = task.get("retry_count", 0) + 1
                task["retried_at"] = utc_now_iso()
                retry_version = task["state_version"]

            # I/O outside lock
            if not save_single_task(task_id, "admin_retry"):
                _restore_task_snapshot_if_unchanged(
                    task_id,
                    snapshot,
                    expected_status=TaskState.FILE_READY.value,
                    expected_version=retry_version,
                )
                return jsonify({"error": "Internal server error"}), 500
            _invalidate_admin_cache()

            return jsonify(
                {
                    "success": True,
                    "message": f"Task {task_id} reset to file_ready for retry",
                }
            )
        except Exception:
            logger.exception("Failed to retry task %s", task_id)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/admin/export-tasks")
    @require_admin
    def export_tasks():
        """Export tasks to CSV"""
        try:
            import csv
            from io import StringIO

            # Get filter parameters
            status = request.args.get("status", "")
            time_range = request.args.get("timeRange", "")

            # Merge DB history with in-memory state so export covers all tasks
            merged = get_all_tasks_merged()

            tasks_to_export = []
            for task_id, task in merged.items():
                if status and task.get("status") != status:
                    continue

                if time_range:
                    try:
                        created_at = parse_timestamp(task.get("created_at", ""))
                    except (ValueError, TypeError):
                        continue
                    now = utc_now()
                    time_limits = {
                        "1h": timedelta(hours=1),
                        "24h": timedelta(days=1),
                        "7d": timedelta(days=7),
                        "30d": timedelta(days=30),
                    }
                    if (
                        time_range in time_limits
                        and now - created_at > time_limits[time_range]
                    ):
                        continue

                task_copy = task.copy()
                task_copy["id"] = task_id
                tasks_to_export.append(task_copy)

            # Create CSV
            output = StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow(
                [
                    "Task ID",
                    "IP Address",
                    "Status",
                    "Models",
                    "Created At",
                    "Completed At",
                    "Error",
                    "Filename",
                ]
            )

            # Data
            for task in tasks_to_export:
                models = task.get("models", [])
                models_str = (
                    ", ".join(models) if isinstance(models, list) else str(models)
                )
                writer.writerow(
                    [
                        task["id"],
                        task.get("ip_address", ""),
                        task.get("status", ""),
                        _sanitize_for_csv(models_str),
                        task.get("created_at", ""),
                        task.get("completed_at", ""),
                        _sanitize_for_csv(str(task.get("error", ""))),
                        _sanitize_for_csv(str(task.get("filename", ""))),
                    ]
                )

            # Return CSV file
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=tasks_export_{utc_now().strftime('%Y%m%d_%H%M%S')}.csv"
                },
            )
        except Exception:
            logger.exception("Failed to export tasks")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api-documentation")
    def api_documentation():
        """API documentation page"""
        return render_template("api_documentation.html")

    # Chinese language routes — redirect to main site (language handled by frontend switcher)
    @app.route("/zh/")
    @app.route("/zh/<path:subpath>")
    def chinese_version(**_route_values):
        """Redirect legacy /zh/ URLs to main site"""
        return redirect(url_for("index"), code=301)

    # Sitemap route
    @app.route("/sitemap.xml")
    def sitemap():
        """Serve sitemap.xml with correct content type"""
        return send_file("static/sitemap.xml", mimetype="application/xml")

    # Robots.txt route
    @app.route("/robots.txt")
    def robots():
        """Serve robots.txt"""
        return send_file("static/robots.txt", mimetype="text/plain")

    # 404 Error Handler
    @app.errorhandler(413)
    def request_entity_too_large(e):
        """Handle file uploads exceeding MAX_CONTENT_LENGTH"""
        return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413

    @app.errorhandler(404)
    def page_not_found(e):
        """Custom 404 page"""
        return render_template("404.html"), 404

    return app


def convert_dataframe_to_marker_genes(df: pd.DataFrame) -> dict[str, list[str]]:
    """Convert supported wide, column-list, or long marker tables to a mapping."""

    def normalize_gene(value):
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                return None
        except (TypeError, ValueError):
            pass
        if value is None or isinstance(value, (bool, int, float)):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            float(text)
            return None
        except (ValueError, OverflowError):
            return text

    def cluster_label(value):
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                return None
        except (TypeError, ValueError):
            pass
        if value is None or isinstance(value, bool):
            return None
        label = str(value).strip()
        if not label:
            return None
        return label if label.lower().startswith("cluster") else f"Cluster_{label}"

    def append_genes(target, cluster, values):
        genes = target.setdefault(cluster, [])
        seen = set(genes)
        for value in values:
            gene = normalize_gene(value)
            if gene and gene not in seen:
                genes.append(gene)
                seen.add(gene)

    marker_genes = {}
    cluster_columns = [
        column for column in df.columns if str(column).strip().lower() == "cluster"
    ]

    if cluster_columns:
        cluster_column = cluster_columns[0]
        value_columns = [column for column in df.columns if column != cluster_column]
        normalized_names = {
            str(column).strip().lower().replace(" ", "_"): column
            for column in value_columns
        }
        gene_column = next(
            (
                normalized_names[name]
                for name in (
                    "gene",
                    "genes",
                    "gene_symbol",
                    "marker",
                    "marker_gene",
                )
                if name in normalized_names
            ),
            None,
        )

        for _, row in df.iterrows():
            cluster = cluster_label(row[cluster_column])
            if cluster is None:
                continue
            values = (
                [row[gene_column]]
                if gene_column
                else [row[col] for col in value_columns]
            )
            append_genes(marker_genes, cluster, values)
    else:
        for column in df.columns:
            cluster = str(column).strip()
            if cluster:
                append_genes(marker_genes, cluster, df[column].tolist())

    marker_genes = {cluster: genes for cluster, genes in marker_genes.items() if genes}
    if not marker_genes:
        raise ValueError(
            "No gene symbols were found. Cells must contain marker-gene names, not numeric expression values."
        )
    return marker_genes


def _annotation_error_message(error, api_keys):
    """Return a bounded error message with submitted credentials redacted."""
    message = str(error).strip() or type(error).__name__
    for api_key in api_keys.values():
        if api_key:
            message = message.replace(api_key, "[redacted]")
    return message[:2000]


def _buffer_task_heartbeat(task_id, run_id, state_version):
    """Record a run-scoped heartbeat for the next persistence flush."""
    with HEARTBEAT_BUFFER_LOCK:
        HEARTBEAT_BUFFER[task_id] = (run_id, state_version, utc_now_iso())


def _fail_annotation_task(task_id, error_message, *, expected_run_id=None):
    """Fail a task only when the error belongs to its current execution."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            logger.error(
                "Annotation task %s disappeared before failure handling", task_id
            )
            return
        if expected_run_id is not None and task.get("run_id") != expected_run_id:
            logger.info(
                "Ignoring failure from superseded run %s for task %s",
                expected_run_id,
                task_id,
            )
            return
        if task.get("status") in TERMINAL_STATES:
            return
        try:
            TaskStateMachine(task).transition_to(TaskState.FAILED, error_message)
        except TaskStateError as state_error:
            logger.error("Cannot fail task %s: %s", task_id, state_error)
            return
    _bg_save(task_id, "task_failed")


def process_annotation(
    task_id,
    run_id,
    species,
    tissue,
    models,
    api_keys,
    consensus_threshold,
    entropy_threshold,
    max_rounds,
    consensus_model,
):
    """Run one annotation job and persist its terminal state."""
    heartbeat_updater = None
    try:
        if not ensure_task_loaded(task_id):
            logger.error("Annotation task %s was not found", task_id)
            return

        with TASKS_LOCK:
            task = TASKS[task_id]
            if (
                task.get("status") != TaskState.QUEUED.value
                or task.get("run_id") != run_id
            ):
                logger.info(
                    "Skipping superseded run %s for task %s in state %s",
                    run_id,
                    task_id,
                    task.get("status"),
                )
                return
            TaskStateMachine(task).transition_to(TaskState.PROCESSING)
            dataframe = task.get("dataframe")

        _bg_save(task_id, "processing_started")
        if not isinstance(dataframe, pd.DataFrame):
            raise RuntimeError("Original marker-gene data is unavailable")

        marker_genes = convert_dataframe_to_marker_genes(dataframe)
        total_clusters = len(marker_genes)
        model_list = []
        api_key_dict = {}
        for model in models:
            provider, model_name = model.split(":", 1)
            model_list.append({"provider": provider, "model": model_name})
            api_key_dict[provider] = api_keys[provider]

        consensus_model_spec = None
        if consensus_model:
            provider, model_name = consensus_model.split(":", 1)
            consensus_model_spec = {"provider": provider, "model": model_name}

        with TASKS_LOCK:
            task = TASKS.get(task_id)
            if (
                task is None
                or task.get("status") != TaskState.PROCESSING.value
                or task.get("run_id") != run_id
            ):
                logger.info(
                    "Stopping superseded run %s before provider calls for task %s",
                    run_id,
                    task_id,
                )
                return
            task["progress"] = {
                "current": 0,
                "total": total_clusters,
                "stage": "Running consensus annotation",
                "percentage": 5,
                "message": f"Processing {total_clusters} clusters",
            }
            state_version = task.get("state_version", 0)
        _buffer_task_heartbeat(task_id, run_id, state_version)

        heartbeat_updater = HeartbeatUpdater(task_id, run_id, interval=30)
        heartbeat_updater.start()
        try:
            results = interactive_consensus_annotation(
                marker_genes=marker_genes,
                species=species,
                models=model_list,
                api_keys=api_key_dict,
                tissue=tissue or None,
                consensus_threshold=consensus_threshold,
                entropy_threshold=entropy_threshold,
                max_discussion_rounds=max_rounds,
                consensus_model=consensus_model_spec,
                verbose=False,
                use_cache=True,
            )
        finally:
            heartbeat_updater.stop()
            heartbeat_updater = None

        flush_heartbeat_buffer()
        formatted_results = format_results_for_web(results)

        with TASKS_LOCK:
            task = TASKS.get(task_id)
            if task is None:
                logger.error(
                    "Annotation task %s disappeared before completion", task_id
                )
                return
            if task.get("status") != TaskState.PROCESSING.value:
                logger.info(
                    "Discarding results for task %s in state %s",
                    task_id,
                    task.get("status"),
                )
                return
            if task.get("run_id") != run_id:
                logger.info(
                    "Discarding results from superseded run %s for task %s",
                    run_id,
                    task_id,
                )
                return

            TaskStateMachine(task).transition_to(TaskState.COMPLETED)
            task["results"] = formatted_results
            task["progress"] = {
                "current": total_clusters,
                "total": total_clusters,
                "stage": "Completed",
                "percentage": 100,
                "message": "Annotation completed successfully",
            }

        if not _bg_save(task_id, "task_completed", retries=2):
            with TASKS_LOCK:
                if (
                    task_id in TASKS
                    and TASKS[task_id].get("run_id") == run_id
                    and TASKS[task_id].get("status") == TaskState.COMPLETED.value
                ):
                    TASKS[task_id]["persistence_failed"] = True

        if MEMORY_MANAGEMENT_AVAILABLE:
            with TASKS_LOCK:
                cleanup_completed_task_dataframe(TASKS[task_id])
                should_cleanup, reason = should_trigger_cleanup(TASKS)
                cleanup_results = (
                    perform_memory_cleanup(TASKS) if should_cleanup else None
                )
            if cleanup_results:
                logger.info(
                    "Memory cleanup after task %s: %s", task_id, cleanup_results
                )
            elif should_cleanup:
                logger.info(
                    "Memory cleanup requested after task %s: %s", task_id, reason
                )

        logger.info(
            "Annotation task %s completed with %s consensus annotations",
            task_id,
            len(formatted_results.get("consensus", {})),
        )
    except Exception as error:
        error_message = _annotation_error_message(error, api_keys)
        logger.error(
            "Annotation task %s failed (%s): %s",
            task_id,
            type(error).__name__,
            error_message,
        )
        flush_heartbeat_buffer()
        _fail_annotation_task(
            task_id,
            error_message,
            expected_run_id=run_id,
        )
    finally:
        if heartbeat_updater is not None:
            heartbeat_updater.stop()
        _annotation_slots.release()


def format_results_for_web(results):
    """Preserve supported result fields and convert them to strict JSON values."""
    if not isinstance(results, dict):
        raise ValueError("Annotation results must be a dictionary")
    return to_json_compatible(
        {
            "consensus": results.get("consensus", {}),
            "consensus_proportion": results.get("consensus_proportion", {}),
            "entropy": results.get("entropy", {}),
            "controversial_clusters": results.get("controversial_clusters", []),
            "metadata": results.get("metadata", {}),
            "discussion_logs": results.get("discussion_logs", {}),
            "model_annotations": results.get("model_annotations", {}),
            "resolved": results.get("resolved", {}),
            "processing_details": results.get("processing_details", {}),
        }
    )


def _finite_float(value, default=0.0):
    """Convert a value to a finite float without breaking downloads or logs."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def create_results_dataframe(results):
    """Create DataFrame for download"""
    if not results:
        raise ValueError("No results are available")
    if not isinstance(results, dict):
        raise ValueError("Results must be a dictionary")

    # Extract data from results
    consensus = results.get("consensus", {})
    consensus_proportion = results.get("consensus_proportion", {})
    entropy = results.get("entropy", {})

    if not consensus:
        raise ValueError("Results do not contain consensus annotations")

    data = []
    for cluster_id in consensus.keys():
        cell_type = consensus.get(cluster_id, "Unknown")
        score = _finite_float(consensus_proportion.get(cluster_id, 0))
        entropy_val = _finite_float(entropy.get(cluster_id, 0))

        data.append(
            {
                "cluster": _sanitize_for_csv(str(cluster_id)),
                "cell_type": _sanitize_for_csv(str(cell_type)),
                "consensus_score": score,
                "entropy": entropy_val,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by cluster name for consistency
    if not df.empty:
        df = df.sort_values("cluster")

    return df


def create_annotation_log_content(results, task):
    """Create human-readable annotation log content"""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("mLLMCelltype Annotation Logs")
    lines.append("=" * 80)
    lines.append("")

    # Task information
    lines.append("TASK INFORMATION:")
    lines.append("-" * 40)
    lines.append(f"Task ID: {task.get('id', 'N/A')}")
    lines.append(f"Created: {task.get('created_at', 'N/A')}")
    lines.append(f"Species: {task.get('species', 'N/A')}")
    lines.append(f"Tissue: {task.get('tissue', 'N/A') or 'Not specified'}")
    lines.append(f"Models: {', '.join(task.get('models', []))}")
    lines.append("")

    # Consensus results
    consensus = results.get("consensus", {})
    consensus_proportion = results.get("consensus_proportion", {})
    entropy = results.get("entropy", {})

    lines.append("CONSENSUS RESULTS:")
    lines.append("-" * 40)
    for cluster_id in consensus.keys():
        cell_type = consensus.get(cluster_id, "Unknown")
        score = _finite_float(consensus_proportion.get(cluster_id, 0))
        entropy_val = _finite_float(entropy.get(cluster_id, 0))
        lines.append(
            f"{cluster_id}: {cell_type} (confidence: {score:.3f}, entropy: {entropy_val:.3f})"
        )
    lines.append("")

    # Model annotations
    model_annotations = results.get("model_annotations", {})
    if model_annotations:
        lines.append("INDIVIDUAL MODEL ANNOTATIONS:")
        lines.append("-" * 40)
        for cluster_id, annotations in model_annotations.items():
            lines.append(f"\n{cluster_id}:")
            for model, annotation in annotations.items():
                lines.append(f"  {model}: {annotation}")
        lines.append("")

    # Controversial clusters
    controversial = results.get("controversial_clusters", [])
    if controversial:
        lines.append("CONTROVERSIAL CLUSTERS:")
        lines.append("-" * 40)
        for cluster in controversial:
            lines.append(f"- {cluster}")
        lines.append("")

    # Discussion logs
    discussion_logs = results.get("discussion_logs", {})
    if discussion_logs:
        lines.append("DISCUSSION LOGS:")
        lines.append("-" * 40)
        for cluster_id, logs in discussion_logs.items():
            lines.append(f"\n=== Discussion for {cluster_id} ===")
            if isinstance(logs, list):
                for i, log_entry in enumerate(logs, 1):
                    lines.append(f"\nRound {i}:")
                    lines.append(str(log_entry))
            else:
                lines.append(str(logs))
        lines.append("")

    # Resolved clusters
    resolved = results.get("resolved", {})
    if resolved:
        lines.append("RESOLVED CLUSTERS:")
        lines.append("-" * 40)
        for cluster_id, resolution in resolved.items():
            lines.append(f"\n{cluster_id}:")
            lines.append(f"  Final decision: {resolution}")
        lines.append("")

    # Metadata
    metadata = results.get("metadata", {})
    if metadata:
        lines.append("PROCESSING METADATA:")
        lines.append("-" * 40)
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("End of annotation logs")
    lines.append("=" * 80)

    return "\n".join(lines)


def cleanup():
    """Cleanup function called on app shutdown.

    Order matters: stop background threads first, then flush and save.
    """
    logger.info("Stopping background threads")
    memory_cleanup_thread.stop()
    task_monitoring_thread.stop()
    flush_heartbeat_buffer()
    save_tasks()
    logger.info("Shutdown persistence completed")


# Register cleanup function
atexit.register(cleanup)

# Create the application
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
