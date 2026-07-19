"""MiniMax provider module for LLMCellType."""

from __future__ import annotations

import re
from typing import Any

import requests

from ..logger import write_log
from ..url_utils import get_working_minimax_endpoint
from .common import (
    NonRetryableProviderError,
    RetryableProviderError,
    UsageSink,
    build_chat_completions_body,
    call_openai_compatible_api,
    ensure_api_key,
    normalize_response_lines,
    resolve_endpoint_url,
)

# MiniMax reports business errors via HTTP 200 + base_resp.status_code. These
# codes are transient per the MiniMax docs (1000 unknown error, 1001 timeout,
# 1002 rate limit, 1039 token limit) and should be retried; all other non-zero
# codes (1004 auth, 1008 balance, 1027 content policy, 2013 invalid params) are
# fatal.
_MINIMAX_TRANSIENT_STATUS_CODES = frozenset({1000, 1001, 1002, 1039})


def _parse_minimax_response(content: dict[str, Any]) -> list[str]:
    """Parse MiniMax response while honoring base_resp business status."""
    base_resp = content.get("base_resp")
    if isinstance(base_resp, dict):
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0):
            error_msg = base_resp.get("status_msg") or "Unknown MiniMax error"
            # Coerce string/float wire codes to int (the R port does the same)
            # so a transient code delivered as "1002" is not misread as fatal.
            try:
                code = int(status_code)
            except (TypeError, ValueError):
                code = status_code
            if code in _MINIMAX_TRANSIENT_STATUS_CODES:
                raise RetryableProviderError(f"MiniMax API error: {error_msg}")
            raise NonRetryableProviderError(f"MiniMax API error: {error_msg}")

    try:
        response_content = content["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected response format from MiniMax: {content}") from e

    if not isinstance(response_content, str):
        return normalize_response_lines(response_content, "MiniMax")

    # Strip <think>...</think> reasoning block (MiniMax coding models)
    response_content = re.sub(r"<think>[\s\S]*?</think>\s*", "", response_content)
    return normalize_response_lines(response_content, "MiniMax")


def process_minimax(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
) -> list[str]:
    """Process request using MiniMax models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed', 'MiniMax-M2.5')
        api_key: MiniMax API key
        base_url: Optional custom base URL
        usage_sink: Optional dict populated in place with token usage.

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting MiniMax API request with model: {model}")

    api_key = ensure_api_key(api_key, "MiniMax")

    # Use custom URL or smart selection
    if base_url:
        url = resolve_endpoint_url("minimax", "MiniMax", base_url)
    else:
        url = get_working_minimax_endpoint(api_key)
        write_log(f"Using smart-selected endpoint: {url}")

    write_log(f"Using model: {model}")
    write_log(f"API URL: {url}")

    body = build_chat_completions_body(
        model=model,
        prompt=prompt,
        user_name="user",
    )
    # Keep only minimal MiniMax-specific behavior:
    # 1) endpoint auto-selection
    # 2) base_resp business-status check in parser
    return call_openai_compatible_api(
        provider_name="MiniMax",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
        response_parser=_parse_minimax_response,
        non_retry_exceptions=(NonRetryableProviderError,),
        usage_sink=usage_sink,
    )
