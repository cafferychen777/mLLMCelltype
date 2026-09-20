"""LiteLLM gateway provider module for LLMCellType.

LiteLLM is a self-hosted AI gateway that exposes a single OpenAI-compatible API
in front of 100+ model providers, adding centralized cost tracking, budgets,
rate limiting, fallbacks, and load balancing. Pointing mLLMCelltype at a gateway
lets a lab route every annotation and discussion round through one endpoint and
one key, while still mixing models from different vendors in a consensus run.

Models are addressed with a ``litellm/`` prefix, for example ``litellm/gpt-5.5``
or ``litellm/claude-opus-4-7``. The prefix selects this provider; whatever
follows it is passed to the gateway untouched, so it can be any name the gateway
routes, including one of its own aliases.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from ..logger import write_log
from .common import (
    UsageSink,
    build_chat_completions_body,
    call_openai_compatible_api,
    resolve_endpoint_url,
)

# Model names are namespaced so the gateway can be told apart from a bare
# provider model name. Everything after the prefix belongs to the gateway.
MODEL_PREFIX = "litellm/"


def strip_model_prefix(model: str) -> str:
    """Return the gateway-facing model name, without the routing prefix.

    The ``litellm/`` prefix exists only so mLLMCelltype can route to this
    provider. The gateway itself knows nothing about it, so it is removed before
    the request is built.
    """
    if model.lower().startswith(MODEL_PREFIX):
        return model[len(MODEL_PREFIX) :]
    return model


def resolve_gateway_key(api_key: str | None) -> str:
    """Resolve the gateway key, allowing an unauthenticated gateway.

    Unlike the vendor providers, a LiteLLM gateway started without a master key
    serves unauthenticated requests, which is the common setup for a gateway
    bound to localhost. A missing key is therefore not an error here, so this
    deliberately does not use :func:`ensure_api_key`.

    Note this is the gateway's master or virtual key, not an upstream provider
    key. Upstream credentials live server-side in the gateway's own config.
    """
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    env_key = os.environ.get("LITELLM_API_KEY", "")
    if env_key.strip():
        return env_key.strip()

    write_log(
        "No LiteLLM API key provided; sending an unauthenticated request. "
        "This works for a gateway started without a master key.",
        level="warning",
    )
    return ""


def models_url(chat_url: str) -> str:
    """Derive the gateway's ``/models`` URL from its chat completions URL."""
    suffix = "/chat/completions"
    if chat_url.endswith(suffix):
        return f"{chat_url[: -len(suffix)]}/models"
    return f"{chat_url.rstrip('/')}/models"


def list_litellm_models(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> list[str]:
    """List the models the configured gateway is currently serving.

    The reachable model set is whatever the gateway operator configured, so it
    is discovered rather than hardcoded. Useful for checking which models are
    available before assembling a consensus run.

    Args:
        api_key: Gateway key. Falls back to ``LITELLM_API_KEY``, then to an
            unauthenticated request.
        base_url: Optional custom gateway URL.
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
    """Process request through a LiteLLM gateway.

    Args:
        prompt: The prompt to send to the gateway
        model: The model name, with or without the ``litellm/`` prefix
            (e.g. ``'litellm/gpt-5.5'``, ``'litellm/claude-opus-4-7'``)
        api_key: Gateway master or virtual key. May be empty for a gateway
            started without a master key.
        base_url: Optional custom gateway URL. Defaults to a local gateway.
        usage_sink: Optional dict populated in place with token usage reported
            by the gateway.
        normalize_response: Whether to split the reply into cleaned lines.

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting LiteLLM gateway request with model: {model}")

    resolved_key = resolve_gateway_key(api_key)
    url = resolve_endpoint_url("litellm", "LiteLLM", base_url)
    gateway_model = strip_model_prefix(model)

    if not gateway_model:
        raise ValueError(f"No model name left after the '{MODEL_PREFIX}' prefix: {model!r}")

    write_log(f"Routing to gateway model: {gateway_model}")

    body = build_chat_completions_body(model=gateway_model, prompt=prompt)

    return call_openai_compatible_api(
        provider_name="LiteLLM",
        api_key=resolved_key,
        url=url,
        body=body,
        post_func=requests.post,
        usage_sink=usage_sink,
        normalize_response=normalize_response,
    )
