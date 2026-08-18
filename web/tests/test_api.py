"""Integration tests for the public task API."""

import io
import importlib.metadata
import uuid

import pytest

import app as app_module


@pytest.fixture(autouse=True)
def clear_tasks() -> None:
    with app_module.TASKS_LOCK:
        app_module.TASKS.clear()
    app_module.storage_manager.tasks.clear()


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True, RATELIMIT_ENABLED=False)
    return app_module.app.test_client()


def upload_markers(client) -> str:
    response = client.post(
        "/api/upload",
        data={
            "file": (
                io.BytesIO(b"Cluster 0,Cluster 1\nCD3D,MS4A1\nCD3E,CD79A\n"),
                "markers.csv",
            )
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    return response.get_json()["task_id"]


def authenticate_admin(client, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "ADMIN_ENABLED", True)
    with client.session_transaction() as session:
        session["is_admin"] = True


def test_upload_and_status_require_owning_session(client) -> None:
    task_id = upload_markers(client)

    assert client.get(f"/api/tasks/{task_id}").status_code == 200

    with client.session_transaction() as session:
        session.pop("owned_task_ids", None)
    assert client.get(f"/api/tasks/{task_id}").status_code == 200

    other_client = app_module.app.test_client()
    response = other_client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 403


def test_annotation_rejects_non_finite_threshold(client) -> None:
    task_id = upload_markers(client)
    response = client.post(
        "/api/annotate",
        json={
            "task_id": task_id,
            "species": "human",
            "tissue": "blood",
            "models": ["openai:gpt-4.1"],
            "api_keys": {"openai": "test-key"},
            "consensusThreshold": "nan",
        },
    )

    assert response.status_code == 400
    assert "consensusThreshold" in response.get_json()["error"]


def test_annotation_rechecks_task_state_after_acquiring_worker_slot(
    client, monkeypatch
) -> None:
    task_id = upload_markers(client)

    class RacingSlot:
        def __init__(self) -> None:
            self.releases = 0

        def acquire(self, blocking=False) -> bool:
            assert blocking is False
            with app_module.TASKS_LOCK:
                app_module.TASKS[task_id]["status"] = "queued"
            return True

        def release(self) -> None:
            self.releases += 1

    slot = RacingSlot()
    monkeypatch.setattr(app_module, "_annotation_slots", slot)

    response = client.post(
        "/api/annotate",
        json={
            "task_id": task_id,
            "species": "mouse",
            "models": ["openai:gpt-4.1"],
            "api_keys": {"openai": "test-key"},
        },
    )

    assert response.status_code == 409
    assert slot.releases == 1
    with app_module.TASKS_LOCK:
        assert "species" not in app_module.TASKS[task_id]


def test_failed_queue_save_does_not_overwrite_concurrent_cancellation(
    client, monkeypatch
) -> None:
    task_id = upload_markers(client)

    def cancel_then_fail_save(saved_task_id, reason="manual") -> bool:
        assert saved_task_id == task_id
        assert reason == "queued"
        with app_module.TASKS_LOCK:
            app_module.TaskStateMachine(app_module.TASKS[task_id]).transition_to(
                app_module.TaskState.CANCELLED,
                "Concurrent cancellation",
            )
        return False

    monkeypatch.setattr(app_module, "save_single_task", cancel_then_fail_save)

    response = client.post(
        "/api/annotate",
        json={
            "task_id": task_id,
            "species": "human",
            "models": ["openai:gpt-4.1"],
            "api_keys": {"openai": "test-key"},
        },
    )

    assert response.status_code == 500
    with app_module.TASKS_LOCK:
        assert app_module.TASKS[task_id]["status"] == "cancelled"


def test_database_load_does_not_overwrite_newer_memory_state(monkeypatch) -> None:
    task_id = str(uuid.uuid4())

    class RacingStorage:
        def get_task(self, requested_task_id):
            assert requested_task_id == task_id
            with app_module.TASKS_LOCK:
                app_module.TASKS[task_id] = {
                    "status": "queued",
                    "state_version": 2,
                }
            return {"status": "file_ready", "state_version": 1}

    monkeypatch.setattr(app_module, "storage_manager", RacingStorage())

    assert app_module.load_single_task(task_id) is True
    with app_module.TASKS_LOCK:
        assert app_module.TASKS[task_id] == {
            "status": "queued",
            "state_version": 2,
        }


def test_provider_defaults_come_from_annotation_package(client) -> None:
    response = client.get("/api/provider-defaults")

    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload["providers"]) == app_module.SUPPORTED_PROVIDERS
    defaults = payload["defaults"]
    assert defaults["openai"]
    assert defaults["anthropic"]
    assert defaults["kimi"] == "kimi-k2.6"


def test_health_checks_annotation_engine(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "healthy",
        "annotation_engine": "available",
    }


def test_deployment_info_reports_annotation_engine_version(client) -> None:
    response = client.get("/api/deployment-info")

    assert response.status_code == 200
    assert response.get_json()["annotation_engine_version"] == (
        importlib.metadata.version("mllmcelltype")
    )


def test_production_app_requires_stable_session_secret(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "LOCAL_TESTING", False)
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY is required"):
        app_module.create_app()


def test_api_key_test_rejects_unknown_provider_without_network_call(client) -> None:
    response = client.post(
        "/api/test-api-key",
        json={"provider": "unknown", "api_key": "test-key"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Unsupported provider"


def test_processing_status_accepts_legacy_numeric_progress(client) -> None:
    task_id = str(uuid.uuid4())
    with app_module.TASKS_LOCK:
        app_module.TASKS[task_id] = {
            "status": "processing",
            "progress": 42,
            "owner_session_required": True,
        }
    with client.session_transaction() as session:
        session["owned_task_ids"] = [task_id]

    response = client.get(f"/api/tasks/{task_id}")

    assert response.status_code == 200
    assert response.get_json()["progress"] == 42


def test_admin_detail_accepts_legacy_numeric_progress(client, monkeypatch) -> None:
    authenticate_admin(client, monkeypatch)
    task_id = str(uuid.uuid4())
    with app_module.TASKS_LOCK:
        app_module.TASKS[task_id] = {
            "status": "processing",
            "progress": 42,
            "created_at": "2026-01-01T00:00:00+00:00",
        }

    response = client.get(f"/api/admin/task/{task_id}")

    assert response.status_code == 200
    assert response.get_json()["progress"] == 42


def test_cancel_does_not_resurrect_task_when_persistence_fails(
    client, monkeypatch
) -> None:
    authenticate_admin(client, monkeypatch)
    monkeypatch.setattr(app_module, "_bg_save", lambda *args, **kwargs: False)
    task_id = str(uuid.uuid4())
    with app_module.TASKS_LOCK:
        app_module.TASKS[task_id] = {
            "status": "processing",
            "state_version": 1,
        }

    response = client.post(f"/api/admin/tasks/{task_id}/cancel")

    assert response.status_code == 202
    with app_module.TASKS_LOCK:
        task = app_module.TASKS[task_id]
        assert task["status"] == "cancelled"
        assert task["persistence_failed"] is True


def test_admin_can_delete_file_ready_task(client, monkeypatch) -> None:
    authenticate_admin(client, monkeypatch)
    task_id = str(uuid.uuid4())
    task = {"status": "file_ready", "state_version": 1}
    with app_module.TASKS_LOCK:
        app_module.TASKS[task_id] = task
    app_module.storage_manager.save_task(task_id, task)

    response = client.post(
        "/api/admin/bulk-delete",
        json={"task_ids": [task_id]},
    )

    assert response.status_code == 200
    assert response.get_json()["deleted"] == [task_id]
    with app_module.TASKS_LOCK:
        assert task_id not in app_module.TASKS


def test_admin_bulk_delete_reports_missing_task(client, monkeypatch) -> None:
    authenticate_admin(client, monkeypatch)
    task_id = str(uuid.uuid4())

    response = client.post(
        "/api/admin/bulk-delete",
        json={"task_ids": [task_id]},
    )

    assert response.status_code == 207
    assert response.get_json()["failed"] == [task_id]


@pytest.mark.parametrize(
    "path",
    ["/faq", "/scrna-troubleshooting-guide", "/api-documentation"],
)
def test_static_documentation_pages_do_not_require_vue(client, path) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert b"vue.global" not in response.data
    assert b"v-cloak" not in response.data
