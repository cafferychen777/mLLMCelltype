"""URL utilities for base URL resolution."""

from __future__ import annotations

import requests

from .config import get_default_api_url
from .logger import write_log

# Re-export get_default_api_url for backward compatibility
# The actual implementation is in config.py (Single Source of Truth)
__all__ = ["get_default_api_url", "get_working_qwen_endpoint", "resolve_provider_base_url", "validate_base_url"]


def resolve_provider_base_url(provider: str, base_urls: str | dict | None) -> str | None:
    """Resolve provider-specific base URL.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        base_urls: User-provided base URLs (string or dict)

    Returns:
        Resolved base URL or None
    """
    if base_urls is None:
        return None

    if isinstance(base_urls, str):
        return base_urls  # Single URL applies to all providers

    if isinstance(base_urls, dict) and provider in base_urls:
        return base_urls[provider]  # Provider-specific URL

    return None


def validate_base_url(url: str) -> bool:
    """Validate base URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False

    # Basic URL format check
    return url.startswith("http://") or url.startswith("https://")


def get_working_qwen_endpoint(api_key: str) -> str:
    """Smart endpoint selection for Qwen.

    Args:
        api_key: Qwen API key

    Returns:
        Working endpoint URL
    """

    def test_endpoint_connectivity(endpoint: str, api_key: str, timeout: int = 5) -> bool:
        """Test endpoint connectivity.

        Args:
            endpoint: API endpoint URL
            api_key: API key for authentication
            timeout: Timeout in seconds

        Returns:
            True if endpoint is accessible, False otherwise
        """
        try:
            # Send a simple test request
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            test_body = {
                "model": "qwen-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }

            response = requests.post(endpoint, headers=headers, json=test_body, timeout=timeout)

            # Any HTTP response indicates the endpoint is network-reachable
            # 200: success, 400: bad request, 401/403: auth error - all mean endpoint works
            # Only connection failures (timeout, DNS error, etc.) indicate unreachable endpoint
            return response.status_code is not None

        except Exception:
            return False

    endpoints = [
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",  # International
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",  # Domestic (China)
    ]

    write_log("Testing Qwen endpoint connectivity...", level="debug")

    for i, endpoint in enumerate(endpoints):
        endpoint_type = "international" if i == 0 else "domestic"
        write_log(f"Testing {endpoint_type} endpoint: {endpoint}", level="debug")

        if test_endpoint_connectivity(endpoint, api_key):
            write_log(f"{endpoint_type} endpoint is accessible", level="debug")
            return endpoint
        else:
            write_log(f"{endpoint_type} endpoint is not accessible", level="warning")

    # If none are reachable, return international endpoint as fallback
    write_log("No endpoints accessible, using international endpoint as fallback")
    return endpoints[0]
