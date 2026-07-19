"""Anthropic provider module for LLMCellType."""

from __future__ import annotations

from typing import Any

import requests

from ..logger import write_log
from .common import (
    NonRetryableProviderError,
    UsageSink,
    call_http_api_with_retry,
    ensure_api_key,
    normalize_response_lines,
    normalize_usage,
    resolve_endpoint_url,
)

# Model alias mapping: user-friendly names -> official API model IDs
MODEL_ALIASES = {
    # Claude 4.7 series (latest)
    "claude-opus-latest": "claude-opus-4-7",
    # Claude 4.6 series
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-sonnet-latest": "claude-sonnet-4-6",
    # Claude Haiku 4.5 series
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-haiku-latest": "claude-haiku-4-5-20251001",
    # Claude 4.1 series (Aug 2025)
    "claude-opus-4.1": "claude-opus-4-1-20250805",
    # Claude 4 series (May 2025)
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    # Claude 3.7 series (Feb 2025)
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    # Claude 3.5 series (2024)
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-new": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-old": "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
    # Claude 3 series (2024)
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


def _resolve_model_name(model: str) -> str:
    """Resolve model alias to official API model ID."""
    return MODEL_ALIASES.get(model, model)


def _extract_anthropic_text(content: dict[str, Any]) -> str:
    """Concatenate every text block, ignoring non-text (e.g. thinking) blocks.

    Iterating instead of hard-indexing ``content[0]`` keeps the answer intact
    when a response leads with a thinking/reasoning block.
    """
    blocks = content.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"Unexpected response format from Anthropic: {content}")
    text_parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if text is None:
            continue  # non-text block (e.g. thinking / reasoning)
        if not isinstance(text, str):
            raise NonRetryableProviderError(
                "Unexpected non-string response content from Anthropic"
            )
        if block.get("type") in (None, "text"):
            text_parts.append(text)
    if not text_parts:
        raise ValueError(
            f"Unexpected response format from Anthropic: no text block found ({content})"
        )
    return "\n".join(text_parts)


def _parse_anthropic_response(content: dict[str, Any]) -> list[str]:
    """Parse Anthropic response payload into clean lines."""
    text = _extract_anthropic_text(content)
    if content.get("stop_reason") == "max_tokens":
        write_log(
            "Anthropic response was truncated (stop_reason='max_tokens'); "
            "trailing clusters may be marked Unknown",
            level="warning",
        )
    return normalize_response_lines(text, "Anthropic")


def extract_anthropic_usage(content: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize Anthropic input/output token counts to the shared usage schema."""
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


def process_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
) -> list[str]:
    """Process request using Anthropic Claude models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'claude-opus-4-7', 'claude-sonnet-4-6')
        api_key: Anthropic API key
        base_url: Optional custom base URL
        usage_sink: Optional dict populated in place with token usage.

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting Anthropic API request with model: {model}")

    api_key = ensure_api_key(api_key, "Anthropic")

    model = _resolve_model_name(model)
    write_log(f"Using model: {model}")

    url = resolve_endpoint_url("anthropic", "Anthropic", base_url)

    # Prepare request
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
        provider_name="Anthropic",
        url=url,
        body=body,
        headers=headers,
        post_func=requests.post,
        response_parser=_parse_anthropic_response,
        max_retries=3,
        retry_delay=2,
        timeout=30,
        request_json=False,
        usage_sink=usage_sink,
        usage_parser=extract_anthropic_usage,
    )
