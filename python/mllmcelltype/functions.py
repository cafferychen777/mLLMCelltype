from __future__ import annotations

import inspect
from collections.abc import Callable

from . import providers as provider_runtime
from .config import PROVIDER_CONFIGS, get_supported_providers
from .validation import normalize_text

ProviderFunction = Callable[..., list[str] | str]
EXPECTED_PROVIDER_PARAMETERS = ("prompt", "model", "api_key", "base_url", "usage_sink", "normalize_response")

PROVIDER_MODEL_PREFIXES = {
    provider: config.model_prefixes
    for provider, config in PROVIDER_CONFIGS.items()
    if config.model_prefixes
}


def _build_provider_function_registry() -> dict[str, ProviderFunction]:
    """Build and validate provider callables from configuration and runtime exports."""
    configured = set(get_supported_providers())
    implemented = {
        name.removeprefix("process_")
        for name, value in vars(provider_runtime).items()
        if name.startswith("process_") and callable(value)
    }
    if configured != implemented:
        missing = sorted(configured - implemented)
        unexpected = sorted(implemented - configured)
        raise RuntimeError(
            "Provider registry mismatch: "
            f"missing implementations={missing}, unexpected implementations={unexpected}"
        )

    registry: dict[str, ProviderFunction] = {}
    for provider in get_supported_providers():
        provider_func = getattr(provider_runtime, f"process_{provider}")
        try:
            parameters = tuple(inspect.signature(provider_func).parameters)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Cannot inspect provider implementation: {provider}") from error
        if parameters != EXPECTED_PROVIDER_PARAMETERS:
            raise RuntimeError(
                f"Provider signature mismatch for {provider}: "
                f"expected={list(EXPECTED_PROVIDER_PARAMETERS)}, got={list(parameters)}"
            )
        registry[provider] = provider_func
    return registry


# Global provider function mapping retained for orchestration and test injection.
PROVIDER_FUNCTIONS = _build_provider_function_registry()


def get_provider(model: str) -> str:
    """Determine the provider based on the model name.

    Uses prefix matching for efficient provider detection.
    OpenRouter models are identified by the '/' character in the model name.

    Args:
        model: The model name (e.g., 'gpt-5.5', 'claude-opus-4-7', 'anthropic/claude-opus-4.7')

    Returns:
        The provider name (e.g., 'openai', 'anthropic', 'openrouter')

    Raises:
        ValueError: If the provider cannot be determined from the model name
    """
    model_normalized = normalize_text(model, "model", required=True)
    model_lower = model_normalized.lower()

    # Providers whose prefix embeds a '/' must be matched BEFORE the OpenRouter
    # rule below, which otherwise claims every namespaced model name. Without
    # this, 'litellm/gpt-5.5' would be routed to OpenRouter.
    for provider, prefixes in PROVIDER_MODEL_PREFIXES.items():
        for prefix in prefixes:
            if "/" in prefix and model_lower.startswith(prefix.lower()):
                return provider

    # OpenRouter models contain '/' (e.g., 'anthropic/claude-sonnet-4.6')
    if "/" in model_normalized:
        return "openrouter"

    # Match by prefix patterns
    for provider, prefixes in PROVIDER_MODEL_PREFIXES.items():
        for prefix in prefixes:
            if model_lower.startswith(prefix.lower()):
                return provider

    # If no prefix matches, raise an error with helpful message
    supported_prefixes = []
    for provider, prefixes in PROVIDER_MODEL_PREFIXES.items():
        supported_prefixes.extend(f"{p}* ({provider})" for p in prefixes)

    raise ValueError(
        f"Cannot determine provider for model: {model_normalized}. "
        f"Supported model prefixes: {', '.join(supported_prefixes)}"
    )


def validate_provider_model_match(provider: str, model: str, field_name: str) -> None:
    """Reject model names that clearly belong to a different provider.

    Unknown model families remain valid for custom or newly released models.
    OpenRouter and LiteLLM are exempt because they intentionally route models
    from other providers.
    """
    if provider in {"openrouter", "litellm"}:
        return

    try:
        inferred_provider = get_provider(model)
    except ValueError:
        return

    if inferred_provider != provider:
        raise ValueError(
            f"{field_name} provider/model mismatch: "
            f"provider='{provider}', model='{model}' "
            f"(inferred provider '{inferred_provider}')"
        )
