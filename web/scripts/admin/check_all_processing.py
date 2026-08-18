#!/usr/bin/env python3
"""List processing and queued tasks through the admin API."""

import argparse
import getpass
import os
from datetime import datetime

import requests

REQUEST_TIMEOUT_SECONDS = 20


def request_json(session, method, url, **kwargs):
    """Send a bounded request and return its JSON object."""
    try:
        response = session.request(
            method,
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as error:
        raise SystemExit(f"Admin API request failed: {error}") from error
    if not response.ok:
        raise SystemExit(f"Admin API returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as error:
        raise SystemExit("Admin API returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise SystemExit("Admin API returned an unexpected response")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MLLM_WEB_BASE_URL", "http://localhost:8080"),
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("ADMIN_USERNAME", "admin"),
    )
    parser.add_argument("--password", default=os.environ.get("ADMIN_PASSWORD"))
    args = parser.parse_args()
    password = args.password or getpass.getpass("Admin password: ")
    base_url = args.base_url.rstrip("/")

    with requests.Session() as session:
        request_json(
            session,
            "POST",
            f"{base_url}/admin/login",
            json={"username": args.username, "password": password},
        )
        payload = request_json(
            session,
            "GET",
            f"{base_url}/api/admin/all-tasks",
            params={"per_page": 0, "cache": "false"},
        )

    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise SystemExit("Admin API returned an invalid task list")

    processing = [task for task in tasks if task.get("status") == "processing"]
    active = [task for task in tasks if task.get("status") in {"processing", "queued"}]

    print("\n=== PROCESSING TASKS ===")
    for task in processing:
        print(f"\nTask ID: {task.get('id')}")
        print(f"File: {task.get('filename', 'Unknown')}")
        print(f"Created: {task.get('created_at', 'Unknown')}")
        created_at = task.get("created_at")
        if created_at:
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                print(f"Duration: {datetime.now(created.tzinfo) - created}")
            except (AttributeError, TypeError, ValueError):
                print("Duration: unavailable")
    print(f"\nTotal processing tasks: {len(processing)}")

    print("\n=== ACTIVE TASKS IN QUEUE ===")
    for task in active:
        print(f"- {task.get('id')}: {task.get('filename')} ({task.get('status')})")
    print(f"\nTotal active tasks: {len(active)}")


if __name__ == "__main__":
    main()
