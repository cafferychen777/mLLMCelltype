"""Gemini provider module for LLMCellType."""

from __future__ import annotations

import time
from typing import Any

from ..execution import current_execution
from ..logger import write_log
from .common import (
    NonRetryableProviderError,
    UsageSink,
    capture_usage,
    ensure_api_key,
    is_retryable_http_status,
    is_retryable_transport_error,
    normalize_optional_base_url,
    normalize_response_lines,
    normalize_usage,
    prepare_usage_sink,
)

GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_DELAY_SECONDS = 2
GEMINI_TIMEOUT_MILLISECONDS = 30_000


def extract_gemini_usage(response: Any) -> dict[str, Any] | None:
    """Extract token usage from a Gemini SDK response's ``usage_metadata``.

    Returns the shared ``{prompt_tokens, completion_tokens, total_tokens}`` schema,
    or ``None`` when no usage metadata is present. Never raises on absent/odd
    metadata.
    """
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None
    return normalize_usage(
        {
            "prompt_tokens": getattr(metadata, "prompt_token_count", None),
            "completion_tokens": getattr(metadata, "candidates_token_count", None),
            "total_tokens": getattr(metadata, "total_token_count", None),
        }
    )


def _parse_gemini_response(response: Any) -> list[str]:
    """Parse Gemini SDK response into cleaned non-empty lines."""
    response_text = getattr(response, "text", None)
    if response_text is None:
        raise NonRetryableProviderError("Gemini response missing text content")
    return normalize_response_lines(response_text, "Gemini")


def _extract_gemini_raw_text(response: Any) -> str:
    """Return the raw text from a Gemini SDK response."""
    response_text = getattr(response, "text", None)
    if response_text is None:
        raise NonRetryableProviderError("Gemini response missing text content")
    return str(response_text)


def _is_retryable_gemini_error(error: Exception, api_error_type: type[Exception]) -> bool:
    """Apply the shared transient-failure policy to Google SDK errors."""
    if isinstance(error, api_error_type):
        return is_retryable_http_status(getattr(error, "code", None))
    return is_retryable_transport_error(error)


def process_gemini(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
    normalize_response: bool = True,
) -> list[str] | str:
    """Process request using Google Gemini models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite')
        api_key: Google API key
        base_url: Optional custom base URL passed through Google SDK HTTP options
        usage_sink: Optional dict populated in place with token usage.

    Returns:
        List[str]: Processed responses, one per cluster

    Raises:
        ImportError: If google-genai package is not installed

    """
    # Lazy import - only load google-genai when actually using Gemini
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError
    except ImportError as e:
        raise ImportError(
            "Gemini provider requires 'google-genai' package. "
            "Install it with: pip install 'mllmcelltype[gemini]' or pip install google-genai"
        ) from e

    write_log(f"Starting Gemini API request with model: {model}")

    api_key = ensure_api_key(api_key, "Google")
    normalized_base_url = normalize_optional_base_url(base_url, "Gemini")
    prepare_usage_sink(usage_sink)

    run = current_execution()
    http_options_kwargs: dict[str, Any] = {"timeout": GEMINI_TIMEOUT_MILLISECONDS}
    if run is not None:
        http_options_kwargs["timeout"] = max(
            1, int(min(run.request_timeout, run.remaining()) * 1000)
        )
        http_options_kwargs["retry_options"] = types.HttpRetryOptions(attempts=1)
    if normalized_base_url:
        http_options_kwargs["base_url"] = normalized_base_url
        write_log(f"Using custom base URL: {normalized_base_url}")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(**http_options_kwargs),
    )
    write_log(f"Using model: {model}")

    attempts = 1 if run is not None else GEMINI_MAX_ATTEMPTS
    for attempt in range(attempts):
        if run is not None:
            run.remaining()
        try:
            write_log("Sending API request...")

            # Generate content
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=4096),
            )

            result = (
                _parse_gemini_response(response)
                if normalize_response
                else _extract_gemini_raw_text(response)
            )
            capture_usage(response, usage_sink, extract_gemini_usage)
            write_log(f"Got response with {len(result) if isinstance(result, list) else 1} lines")
            write_log(f"Raw response from Gemini:\n{result}", level="debug")
            return result

        except Exception as error:
            if not _is_retryable_gemini_error(error, APIError):
                raise
            write_log(
                f"Error during API call (attempt {attempt + 1}/{GEMINI_MAX_ATTEMPTS}): {error!s}",
                level="error",
            )
            if attempt < attempts - 1:
                wait_time = GEMINI_RETRY_DELAY_SECONDS * (2**attempt)
                rate_limited = isinstance(error, APIError) and getattr(error, "code", None) == 429
                prefix = "Rate limited. " if rate_limited else ""
                write_log(
                    f"{prefix}Waiting {wait_time} seconds before retrying...",
                    level="warning",
                )
                time.sleep(wait_time)
            else:
                raise
