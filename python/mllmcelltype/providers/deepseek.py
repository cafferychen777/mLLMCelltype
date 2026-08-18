"""DeepSeek provider module for LLMCellType."""

from __future__ import annotations

import requests

from ..logger import write_log
from .common import (
    UsageSink,
    build_chat_completions_body,
    call_openai_compatible_api,
    ensure_api_key,
    resolve_endpoint_url,
)


def process_deepseek(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
    normalize_response: bool = True,
) -> list[str] | str:
    """Process request using DeepSeek models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'deepseek-v4-flash', 'deepseek-v4-pro')
        api_key: DeepSeek API key
        base_url: Optional custom base URL
        usage_sink: Optional dict populated in place with token usage.

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting DeepSeek API request with model: {model}")

    api_key = ensure_api_key(api_key, "DeepSeek")
    url = resolve_endpoint_url("deepseek", "DeepSeek", base_url)

    write_log(f"Using model: {model}")

    body = build_chat_completions_body(
        model=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=4096,
    )
    body["thinking"] = {"type": "disabled"}

    return call_openai_compatible_api(
        provider_name="DeepSeek",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
        max_retries=5,
        retry_delay=3,
        timeout=120,
        request_json=True,
        usage_sink=usage_sink,
        normalize_response=normalize_response,
    )
