"""
Rate limiter configuration for Flask application.

Exempts high-frequency read-only endpoints (status polling, health checks)
so the global limit only constrains write/expensive operations.
"""

from flask import request
from flask_limiter import Limiter

from utils.request_context import get_client_ip


def _is_exempt_request() -> bool:
    """Return True to skip rate limiting for lightweight read-only endpoints.

    The frontend polls /api/tasks/<id> every 2 seconds during processing,
    which would exhaust a global 1000/hour limit within minutes.  These
    GET requests are cheap (in-memory dict lookup) and safe to exempt.
    """
    path = request.path
    if path.startswith("/api/tasks/") and request.method == "GET":
        return True
    if path in ("/health", "/api/deployment-info"):
        return True
    return False


def create_limiter(app):
    """Create and configure rate limiter."""
    limiter = Limiter(
        app=app,
        key_func=get_client_ip,
        default_limits=["1000 per hour"],
        storage_uri="memory://",
        headers_enabled=True,
        strategy="fixed-window",
    )

    # Register the exempt filter via the decorator API
    # (request_filter is a method on Limiter, not a constructor param)
    limiter.request_filter(_is_exempt_request)

    return limiter
