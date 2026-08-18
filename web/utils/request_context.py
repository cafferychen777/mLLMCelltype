"""Trusted request metadata shared by API policy and task ownership."""

from flask import request


def get_client_ip() -> str:
    """Return the client IP supplied by the trusted Caddy reverse proxy.

    Production binds the application to loopback, and Caddy overwrites
    ``X-Real-IP`` before forwarding a request. Direct local development falls
    back to Flask's peer address.
    """
    return request.headers.get("X-Real-IP") or request.remote_addr or "unknown"
