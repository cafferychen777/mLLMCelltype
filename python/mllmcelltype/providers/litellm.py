"""LiteLLM provider module for LLMCellType.

[LiteLLM](https://docs.litellm.ai/) gives one interface to 100+ model providers.
It can be used two ways, and this provider supports both from the same model
name:

1. **Direct**, with no extra infrastructure. `litellm/claude-opus-4-7` is routed
   by the LiteLLM SDK straight to Anthropic using `ANTHROPIC_API_KEY`. This is
   what lets a consensus run mix vendors without mLLMCelltype needing a
   dedicated module per vendor.
2. **Through a self-hosted gateway**, when `base_url` (or `LITELLM_API_BASE`) is
   set. The request is then routed with LiteLLM's `litellm_proxy/` prefix, which
   adds centralized cost tracking, budgets, rate limiting, fallbacks, and load
   balancing, and keeps every upstream provider key server-side.

The provider is selected explicitly (``provider="litellm"``), not inferred
from the model name, because LiteLLM routes any vendor's model. The model name
is handed to LiteLLM untouched: ``"claude-opus-4-7"``,
``"anthropic/claude-opus-4-7"``, or any alias a gateway serves.

``litellm`` is an optional dependency, imported lazily so the package installs
and runs without it.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from ..logger import write_log
from .common import (
    NonRetryableProviderError,
    UsageSink,
    capture_usage,
    normalize_response_lines,
    prepare_usage_sink,
    resolve_endpoint_url,
)

# LiteLLM's own prefix for "send this to my gateway rather than routing it
# yourself". See https://docs.litellm.ai/docs/providers/litellm_proxy
PROXY_PREFIX = "litellm_proxy/"

_IMPORT_ERROR_MESSAGE = (
    "The 'litellm' package is required for the litellm provider. "
    "Install it with: pip install 'mllmcelltype[litellm]'"
)


def _import_litellm() -> Any:
    """Import litellm lazily, since it is an optional dependency."""
    try:
        import litellm
    except ImportError as error:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_IMPORT_ERROR_MESSAGE) from error
    return litellm


def resolve_gateway_key(api_key: str | None) -> str:
    """Resolve the gateway/provider key, allowing none to be set.

    Unlike the vendor providers, this one may legitimately run without a key:
    a gateway started without a master key serves unauthenticated requests, and
    in direct mode LiteLLM reads each vendor's own environment variable
    (``ANTHROPIC_API_KEY`` and friends) itself. A missing key is therefore not
    an error here, so this deliberately does not use :func:`ensure_api_key`.
    """
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    env_key = os.environ.get("LITELLM_API_KEY", "")
    if env_key.strip():
        return env_key.strip()

    return ""


def resolve_api_base(base_url: str | None) -> str | None:
    """Resolve the gateway base URL, or ``None`` for direct routing.

    Returns ``None`` when no gateway is configured, which is the signal to let
    LiteLLM route directly to the upstream provider.
    """
    if isinstance(base_url, str) and base_url.strip():
        return base_url.strip().rstrip("/")

    env_base = os.environ.get("LITELLM_API_BASE", "")
    if env_base.strip():
        return env_base.strip().rstrip("/")

    return None


def resolve_sdk_model(model: str, api_base: str | None) -> str:
    """Return the model name LiteLLM should route.

    With a gateway configured the name needs LiteLLM's ``litellm_proxy/``
    prefix, otherwise LiteLLM infers the vendor from the model name and calls
    it directly, silently bypassing the gateway. Without a gateway the bare
    name is what lets LiteLLM route to the vendor.
    """
    if not model:
        raise ValueError("A model name is required for the litellm provider")

    if not api_base or model.startswith(PROXY_PREFIX):
        return model

    return f"{PROXY_PREFIX}{model}"


def _extract_content(response: Any) -> str:
    """Pull the assistant text out of a LiteLLM response."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise NonRetryableProviderError(
            f"Unexpected response format from LiteLLM: {response!r}"
        ) from error

    if not isinstance(content, str):
        raise NonRetryableProviderError("Unexpected non-string response content from LiteLLM")
    return content


def _usage_from_response(response: Any) -> dict[str, Any] | None:
    """Read token usage off a LiteLLM response, tolerating its absence."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif not isinstance(usage, dict):
        usage = {
            field: getattr(usage, field, None)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    if not isinstance(usage, dict):
        return None

    # LiteLLM attaches its computed spend for the call under this hidden param,
    # which is the number a gateway user actually wants to see.
    cost = getattr(response, "_hidden_params", None)
    if isinstance(cost, dict) and cost.get("response_cost") is not None:
        usage = {**usage, "cost": cost["response_cost"]}

    return usage


def models_url(chat_url: str) -> str:
    """Derive a gateway's ``/models`` URL from its chat completions URL."""
    suffix = "/chat/completions"
    if chat_url.endswith(suffix):
        return f"{chat_url[: -len(suffix)]}/models"
    return f"{chat_url.rstrip('/')}/models"


def list_litellm_models(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> list[str]:
    """List the models a configured LiteLLM gateway is serving.

    The reachable model set is whatever the gateway operator configured, so it
    is discovered rather than hardcoded. Useful for checking what is available
    before assembling a consensus run.

    Args:
        api_key: Gateway key. Falls back to ``LITELLM_API_KEY``, then to an
            unauthenticated request.
        base_url: Gateway URL. Falls back to ``LITELLM_API_BASE``, then to the
            configured default.
        timeout: Request timeout in seconds.

    Returns:
        Sorted list of model ids served by the gateway.

    Raises:
        ValueError: If the gateway response cannot be parsed.

    """
    resolved_key = resolve_gateway_key(api_key)
    url = models_url(resolve_endpoint_url("litellm", "LiteLLM", base_url))

    headers = {"Content-Type": "application/json"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    write_log(f"Listing LiteLLM gateway models from: {url}")
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    payload: Any = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError(f"Unexpected model list response from LiteLLM: {payload}")

    model_ids = [entry["id"] for entry in data if isinstance(entry, dict) and entry.get("id")]
    write_log(f"LiteLLM gateway is serving {len(model_ids)} model(s)")
    return sorted(model_ids)


def process_litellm(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    usage_sink: UsageSink | None = None,
    normalize_response: bool = True,
) -> list[str] | str:
    """Process a request through LiteLLM, directly or via a gateway.

    Args:
        prompt: The prompt to send
        model: Model name, with or without the ``litellm/`` prefix
            (e.g. ``'litellm/gpt-5.5'``, ``'litellm/anthropic/claude-opus-4-7'``)
        api_key: Gateway master/virtual key, or the upstream provider key. May be
            empty, in which case LiteLLM reads the vendor's own environment
            variable, or the gateway is called unauthenticated.
        base_url: Gateway URL. When omitted (and ``LITELLM_API_BASE`` is unset),
            LiteLLM routes directly to the upstream provider.
        usage_sink: Optional dict populated in place with token usage, and with
            LiteLLM's computed ``cost`` when it reports one.
        normalize_response: Whether to split the reply into cleaned lines.

    Returns:
        List[str]: Processed responses, one per cluster

    """
    litellm = _import_litellm()

    prepare_usage_sink(usage_sink)

    resolved_key = resolve_gateway_key(api_key)
    api_base = resolve_api_base(base_url)
    sdk_model = resolve_sdk_model(model, api_base)

    if api_base:
        write_log(f"Routing '{sdk_model}' through the LiteLLM gateway at {api_base}")
    else:
        write_log(f"Routing '{sdk_model}' directly with the LiteLLM SDK")

    request: dict[str, Any] = {
        "model": sdk_model,
        "messages": [{"role": "user", "content": prompt}],
        # Providers reject each other's parameters (Anthropic on seed, Gemini on
        # an OpenAI-shaped response_format, and so on). Dropping what a given
        # model does not support is what keeps one prompt usable across every
        # provider in a consensus run.
        "drop_params": True,
    }
    if api_base:
        request["api_base"] = api_base
    if resolved_key:
        request["api_key"] = resolved_key

    try:
        response = litellm.completion(**request)
    except Exception as error:
        # litellm raises OpenAI-compatible exception types. Surface the message
        # as-is; the orchestration layer above decides whether to retry a model.
        write_log(f"LiteLLM request failed: {error!s}", level="error")
        raise

    capture_usage(response, usage_sink, _usage_from_response)

    content = _extract_content(response)
    if not normalize_response:
        return content

    return normalize_response_lines(content, "LiteLLM")
