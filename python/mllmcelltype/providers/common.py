"""Shared helpers for OpenAI-compatible provider implementations."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, MutableMapping
from numbers import Integral, Real
from typing import Any

import requests

from ..logger import write_log
from ..url_utils import get_default_api_url, validate_base_url

ResponseParser = Callable[[dict[str, Any]], list[str]]

# A caller-supplied dict that, when passed, is populated in place with token
# usage for the call: prompt_tokens / completion_tokens / total_tokens, plus an
# optional native `cost` (USD) when the provider returns one. None = no capture.
UsageSink = MutableMapping[str, Any]

# Maps a raw response payload to the normalized usage schema (or None if absent).
UsageParser = Callable[[Any], dict[str, Any] | None]
TOKEN_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _normalize_token_count(value: Any) -> int | None:
    """Normalize one provider token count without accepting bools or estimates."""
    if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def normalize_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize provider telemetry to finite, non-negative canonical fields."""
    normalized = {field: _normalize_token_count(usage.get(field)) for field in TOKEN_USAGE_FIELDS}
    cost = usage.get("cost")
    if isinstance(cost, Real) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
        normalized["cost"] = float(cost)
    return normalized


def extract_chat_completions_usage(content: dict[str, Any]) -> dict[str, Any] | None:
    """Extract token usage from an OpenAI-compatible chat completions payload.

    Returns a normalized ``{prompt_tokens, completion_tokens, total_tokens}`` dict
    (plus ``cost`` when the provider includes a native USD cost, e.g. OpenRouter
    when ``usage: {include: true}`` was requested), or ``None`` when no usage block
    is present. Never raises on a malformed/absent ``usage`` block.

    Resilient to provider format drift by design: only the three canonical keys
    (plus ``cost``) are read; any extra/unknown fields are ignored. Missing keys
    are reported as ``None`` rather than fabricated, so callers can tell "not
    reported" from a real zero. Validated live against OpenRouter; the other
    OpenAI-compatible providers (OpenAI, DeepSeek, Qwen, Grok, StepFun, Zhipu,
    MiniMax) share this exact path but were not run against their native endpoints.
    """
    if not isinstance(content, dict):
        return None
    usage = content.get("usage")
    if not isinstance(usage, dict):
        return None

    return normalize_usage(usage)


class NonRetryableProviderError(ValueError):
    """Provider error that should fail fast without retry."""


class RetryableProviderError(RuntimeError):
    """Provider error that should be retried with the shared backoff.

    Use this for a transient failure a provider signals inside a 200-OK body
    (e.g. MiniMax rate-limit / timeout status codes) so the retry loop treats
    it like a transient HTTP 429/5xx instead of failing fast. It inherits
    RuntimeError so that if it still fails after every retry, the consensus
    layer's RECOVERABLE_LLM_EXCEPTIONS catch (which includes RuntimeError)
    drops just this model instead of aborting the whole multi-model run.
    """


def build_chat_completions_body(
    *,
    model: str,
    prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    user_name: str | None = None,
) -> dict[str, Any]:
    """Build a normalized OpenAI-compatible chat completions request body."""
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if user_name:
        message["name"] = user_name

    body: dict[str, Any] = {
        "model": model,
        "messages": [message],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def normalize_response_lines(response: Any, provider_name: str) -> list[str]:
    """Validate provider text and return cleaned, non-empty response lines."""
    if isinstance(response, str):
        raw_lines = response.splitlines()
    elif isinstance(response, list):
        if not all(isinstance(line, str) for line in response):
            raise NonRetryableProviderError(
                f"Unexpected non-string response content from {provider_name}"
            )
        raw_lines = [nested for line in response for nested in line.splitlines()]
    else:
        raise NonRetryableProviderError(
            f"Unexpected non-string response content from {provider_name}"
        )

    lines = [line.strip().rstrip(",") for line in raw_lines if line.strip().rstrip(",")]
    if not lines:
        raise NonRetryableProviderError(f"Empty response content from {provider_name}")
    if any(line.casefold().startswith("error:") for line in lines):
        raise NonRetryableProviderError(f"Error response content from {provider_name}")
    return lines


def parse_chat_completions_response(content: dict[str, Any], provider_name: str) -> list[str]:
    """Parse OpenAI-compatible chat completions response into clean lines."""
    try:
        text = content["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected response format from {provider_name}: {content}") from e

    if content["choices"][0].get("finish_reason") == "length":
        write_log(
            f"{provider_name} response was truncated (finish_reason='length'); "
            "trailing clusters may be marked Unknown",
            level="warning",
        )
    return normalize_response_lines(text, provider_name)


def _extract_error_message(response: requests.Response) -> str | None:
    """Extract provider error message from a non-200 response."""
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError, TypeError):
        return None

    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            message = error_obj.get("message")
            if isinstance(message, str) and message.strip():
                return message
        if isinstance(error_obj, str) and error_obj.strip():
            return error_obj

        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message

    return None


def ensure_api_key(api_key: str | None, provider_name: str) -> str:
    """Ensure API key is present and return it."""
    if isinstance(api_key, str):
        normalized = api_key.strip()
        if normalized:
            return normalized

    error_msg = f"{provider_name} API key is missing or empty"
    write_log(error_msg, level="error")
    raise ValueError(error_msg)


def normalize_optional_base_url(base_url: str | None, provider_name: str) -> str | None:
    """Normalize and validate an optional provider endpoint override."""
    if base_url is None:
        return None
    if not isinstance(base_url, str):
        raise ValueError(
            f"Invalid base URL type for {provider_name}: "
            f"expected str or None, got {type(base_url).__name__}"
        )
    normalized = base_url.strip().rstrip("/")
    if normalized and not validate_base_url(normalized):
        raise ValueError(f"Invalid base URL: {base_url}")
    return normalized or None


def resolve_endpoint_url(
    provider_key: str,
    provider_name: str,
    base_url: str | None,
) -> str:
    """Resolve endpoint URL with optional custom override and validation."""
    normalized_base_url = normalize_optional_base_url(base_url, provider_name)
    if normalized_base_url:
        write_log(f"Using custom base URL: {normalized_base_url}")
        return normalized_base_url

    default_url = get_default_api_url(provider_key)
    if not default_url:
        raise ValueError(
            f"No default API URL configured for {provider_name} (provider key: {provider_key})"
        )
    write_log(f"Using default URL: {default_url}")
    return default_url


def _build_request_kwargs(
    *,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    request_json: bool,
) -> dict[str, Any]:
    """Build one immutable request payload shared by all retry attempts."""
    kwargs: dict[str, Any] = {"url": url, "headers": headers, "timeout": timeout}
    if request_json:
        kwargs["json"] = body
    else:
        kwargs["data"] = json.dumps(body)
    return kwargs


def _raise_for_provider_status(response: requests.Response, provider_name: str) -> None:
    """Reject every non-200 response with provider error context."""
    if response.status_code == 200:
        return

    error_detail = _extract_error_message(response)
    if error_detail:
        write_log(f"{provider_name} API request failed: {error_detail}", level="error")
    else:
        write_log(
            f"{provider_name} API request failed with status {response.status_code}",
            level="error",
        )

    response.raise_for_status()
    raise requests.HTTPError(
        f"Unexpected HTTP status {response.status_code} from {provider_name}",
        response=response,
    )


def _parse_provider_response(
    content: dict[str, Any],
    provider_name: str,
    response_parser: ResponseParser,
) -> list[str]:
    """Parse and normalize a successful provider payload."""
    try:
        parsed = response_parser(content)
    except (NonRetryableProviderError, RetryableProviderError):
        raise
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise NonRetryableProviderError(
            f"Failed to parse {provider_name} response: {error!s}"
        ) from error

    if not isinstance(parsed, list):
        raise NonRetryableProviderError(
            f"{provider_name} response parser returned {type(parsed).__name__}, expected list"
        )
    return normalize_response_lines(parsed, provider_name)


def _decode_provider_response(response: requests.Response, provider_name: str) -> dict[str, Any]:
    """Decode a successful provider response as the expected JSON object."""
    try:
        content = response.json()
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise NonRetryableProviderError(
            f"Failed to decode {provider_name} response as JSON: {error!s}"
        ) from error
    if not isinstance(content, dict):
        raise NonRetryableProviderError(
            f"Unexpected {provider_name} response type: {type(content).__name__}, expected object"
        )
    return content


def capture_usage(
    content: Any,
    usage_sink: UsageSink | None,
    usage_parser: UsageParser,
) -> None:
    """Capture optional usage without making telemetry part of request success."""
    if usage_sink is None:
        return
    try:
        usage = usage_parser(content)
    except Exception as error:
        write_log(f"Failed to parse token usage: {error!s}", level="warning")
        return
    if usage is None:
        return
    if not isinstance(usage, Mapping):
        write_log(
            f"Ignoring token usage with unexpected type: {type(usage).__name__}",
            level="warning",
        )
        return
    normalized_usage = normalize_usage(usage)
    try:
        usage_sink.update(normalized_usage)
    except Exception as error:
        write_log(f"Failed to store token usage: {error!s}", level="warning")


def prepare_usage_sink(usage_sink: UsageSink | None) -> None:
    """Validate and clear a caller-owned usage sink before an external request."""
    if usage_sink is None:
        return
    if not isinstance(usage_sink, MutableMapping):
        raise ValueError("usage_sink must be a mutable mapping or None")
    try:
        usage_sink.clear()
    except Exception as error:
        raise ValueError(f"usage_sink could not be cleared: {error!s}") from error


def _is_retryable_error(
    error: Exception,
    non_retry_exceptions: tuple[type[Exception], ...],
) -> bool:
    """Return whether a failed attempt may succeed when repeated."""
    if isinstance(error, RetryableProviderError):
        return True
    if isinstance(error, NonRetryableProviderError):
        return False
    if non_retry_exceptions and isinstance(error, non_retry_exceptions):
        return False
    if not isinstance(error, requests.exceptions.HTTPError):
        return is_retryable_transport_error(error)
    if error.response is None:
        return False

    return is_retryable_http_status(error.response.status_code)


def is_retryable_http_status(status_code: Any) -> bool:
    """Return whether an HTTP status represents a transient request failure."""
    return (
        isinstance(status_code, int)
        and not isinstance(status_code, bool)
        and (status_code in {408, 425, 429} or status_code >= 500)
    )


def is_retryable_transport_error(error: Exception) -> bool:
    """Return whether a supported HTTP transport reports a transient failure."""
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True

    try:
        import httpx
    except ImportError:
        return False
    return isinstance(error, httpx.TransportError)


def _validate_retry_settings(max_retries: int, retry_delay: int, timeout: int) -> None:
    """Validate retry settings before issuing any external request."""
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
        raise ValueError("max_retries must be a positive integer")
    if not isinstance(retry_delay, int) or isinstance(retry_delay, bool) or retry_delay < 0:
        raise ValueError("retry_delay must be a non-negative integer")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        raise ValueError("timeout must be a positive integer")


def call_http_api_with_retry(
    *,
    provider_name: str,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    post_func: Callable[..., requests.Response],
    response_parser: ResponseParser,
    max_retries: int = 3,
    retry_delay: int = 2,
    timeout: int = 30,
    request_json: bool = False,
    non_retry_exceptions: tuple[type[Exception], ...] = (),
    usage_sink: UsageSink | None = None,
    usage_parser: UsageParser = extract_chat_completions_usage,
) -> list[str]:
    """Execute an HTTP API request with retry and unified error handling.

    When ``usage_sink`` is provided, stale values are cleared before the call
    and successful response usage is populated via ``usage_parser``. Usage
    parsing is optional telemetry and cannot invalidate a model response.
    """
    _validate_retry_settings(max_retries, retry_delay, timeout)
    prepare_usage_sink(usage_sink)

    write_log("Sending API request...")
    request_kwargs = _build_request_kwargs(
        url=url,
        body=body,
        headers=headers,
        timeout=timeout,
        request_json=request_json,
    )

    for attempt in range(max_retries):
        try:
            response = post_func(**request_kwargs)
            _raise_for_provider_status(response, provider_name)
            content = _decode_provider_response(response, provider_name)
            result = _parse_provider_response(content, provider_name, response_parser)
            capture_usage(content, usage_sink, usage_parser)
            write_log(f"Got response with {len(result)} lines")
            write_log(f"Raw response from {provider_name}:\n{result}", level="debug")
            return result

        except Exception as error:
            if not _is_retryable_error(error, non_retry_exceptions):
                raise

            write_log(
                f"Error during API call (attempt {attempt + 1}/{max_retries}): {error!s}",
                level="error",
            )
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                rate_limited = (
                    isinstance(error, requests.exceptions.HTTPError)
                    and error.response is not None
                    and error.response.status_code == 429
                )
                prefix = "Rate limited. " if rate_limited else ""
                write_log(
                    f"{prefix}Waiting {wait_time} seconds before retrying...",
                    level="warning",
                )
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"{provider_name} API request failed after {max_retries} attempts")


def call_openai_compatible_api(
    *,
    provider_name: str,
    api_key: str,
    url: str,
    body: dict[str, Any],
    post_func: Callable[..., requests.Response],
    response_parser: ResponseParser | None = None,
    extra_headers: dict[str, str] | None = None,
    max_retries: int = 3,
    retry_delay: int = 2,
    timeout: int = 30,
    request_json: bool = False,
    non_retry_exceptions: tuple[type[Exception], ...] = (),
    usage_sink: UsageSink | None = None,
) -> list[str]:
    """Execute a request against an OpenAI-compatible endpoint with retries.

    When ``usage_sink`` is provided, it is populated in place with the call's
    token usage (see :func:`extract_chat_completions_usage`). Default ``None``
    preserves prior behavior exactly.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    parser = response_parser or (
        lambda content: parse_chat_completions_response(content, provider_name)
    )

    return call_http_api_with_retry(
        provider_name=provider_name,
        url=url,
        body=body,
        headers=headers,
        post_func=post_func,
        response_parser=parser,
        max_retries=max_retries,
        retry_delay=retry_delay,
        timeout=timeout,
        request_json=request_json,
        non_retry_exceptions=non_retry_exceptions,
        usage_sink=usage_sink,
    )
