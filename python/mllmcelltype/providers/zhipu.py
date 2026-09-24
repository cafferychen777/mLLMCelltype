"""Zhipu AI (ChatGLM) provider module for LLMCellType."""

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


def process_zhipu(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
    normalize_response: bool = True,
) -> list[str] | str:
    """Process request using Zhipu AI (ChatGLM) models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'glm-5.1', 'glm-5', 'glm-5-turbo')
        api_key: Zhipu AI API key
        base_url: Optional custom base URL
        usage_sink: Optional dict populated in place with token usage.

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting Zhipu AI API request with model: {model}")

    api_key = ensure_api_key(api_key, "Zhipu AI")
    url = resolve_endpoint_url("zhipu", "Zhipu AI", base_url)

    write_log(f"Using model: {model}")

    body = build_chat_completions_body(
        model=model,
        prompt=prompt,
        temperature=0.7,
        # Forced thinking shares the output budget with the final answer.
        max_tokens=16384 if model.lower().startswith("glm-5.3") else 4096,
    )

    return call_openai_compatible_api(
        provider_name="Zhipu AI",
        api_key=api_key,
        url=url,
        body=body,
        post_func=requests.post,
        usage_sink=usage_sink,
        normalize_response=normalize_response,
    )
