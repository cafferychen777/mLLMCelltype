"""Shared test configuration."""

import os


os.environ["LOCAL_TESTING"] = "true"
os.environ["BACKGROUND_THREADS_ENABLED"] = "false"
os.environ["FLASK_SECRET_KEY"] = "test-secret-key"
