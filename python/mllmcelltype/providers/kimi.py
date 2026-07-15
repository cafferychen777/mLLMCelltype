"""Kimi (Moonshot AI) provider module for LLMCellType.

Kimi exposes an Anthropic-compatible Messages API: it uses the same
``/v1/messages`` endpoint, ``x-api-key`` + ``anthropic-version`` headers, and
returns its text under ``content[0]["text"]``.
"""

from __future__ import annotations

from typing import Any

import requests

from ..logger import write_log
from .common import (
    UsageSink,
    call_http_api_with_retry,
    ensure_api_key,
    normalize_response_lines,
    normalize_usage,
    resolve_endpoint_url,
)


def _parse_kimi_response(content: dict[str, Any]) -> list[str]:
    """Parse Kimi (Anthropic-compatible) response payload into clean lines."""
    try:
        text = content["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected response format from Kimi: {content}") from e

    return normalize_response_lines(text, "Kimi")


def extract_kimi_usage(content: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize Kimi input/output token counts to the shared usage schema.

    Kimi follows the Anthropic Messages usage schema
    (``input_tokens``/``output_tokens``).
    """
    usage = content.get("usage")
    if not isinstance(usage, dict):
        return None

    normalized = normalize_usage(
        {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
        }
    )
    prompt_tokens = normalized["prompt_tokens"]
    completion_tokens = normalized["completion_tokens"]
    if prompt_tokens is not None and completion_tokens is not None:
        normalized["total_tokens"] = prompt_tokens + completion_tokens
    return normalized


def process_kimi(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
) -> list[str]:
    """Process request using Kimi (Moonshot AI) models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'kimi-k2.7', 'kimi-k2.7[1m]')
        api_key: Kimi API key
        base_url: Optional custom base URL
        usage_sink: Optional dict populated in place with token usage.

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting Kimi API request with model: {model}")

    api_key = ensure_api_key(api_key, "Kimi")

    url = resolve_endpoint_url("kimi", "Kimi", base_url)

    # Kimi speaks the Anthropic Messages protocol
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }

    return call_http_api_with_retry(
        provider_name="Kimi",
        url=url,
        body=body,
        headers=headers,
        post_func=requests.post,
        response_parser=_parse_kimi_response,
        max_retries=3,
        retry_delay=2,
        timeout=30,
        request_json=False,
        usage_sink=usage_sink,
        usage_parser=extract_kimi_usage,
    )
