#!/usr/bin/env python3
"""Inspect one task through the admin API."""

import argparse
import getpass
import os
from datetime import datetime

import requests

from check_all_processing import request_json


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
    parser.add_argument(
        "--task-id-prefix",
        default=os.environ.get("TASK_ID_PREFIX", ""),
    )
    args = parser.parse_args()
    if not args.task_id_prefix:
        raise SystemExit("Missing --task-id-prefix (or set TASK_ID_PREFIX).")

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

        matches = [
            task
            for task in tasks
            if str(task.get("id", "")).startswith(args.task_id_prefix)
        ]
        if not matches:
            raise SystemExit(f"No task starts with {args.task_id_prefix!r}")
        if len(matches) > 1:
            raise SystemExit(
                f"Task prefix is ambiguous; it matches {len(matches)} tasks"
            )

        task = matches[0]
        task_id = task["id"]
        detail = request_json(
            session,
            "GET",
            f"{base_url}/api/admin/task/{task_id}",
        )

    print(f"Task ID: {task_id}")
    print(f"Status: {detail.get('status', 'Unknown')}")
    print(f"File: {detail.get('filename', 'Unknown')}")
    print(f"Created: {detail.get('created_at', 'Unknown')}")
    print(f"Progress: {detail.get('progress', 0)}%")
    if detail.get("error"):
        print(f"Error: {detail['error']}")

    created_at = detail.get("created_at")
    if created_at:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            print(f"Age: {datetime.now(created.tzinfo) - created}")
        except (AttributeError, TypeError, ValueError):
            print("Age: unavailable")

    logs = detail.get("logs", [])
    if isinstance(logs, list) and logs:
        print("\nRecent logs:")
        for entry in logs[-10:]:
            print(f"  {entry}")


if __name__ == "__main__":
    main()
