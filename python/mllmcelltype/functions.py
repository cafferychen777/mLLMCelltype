from __future__ import annotations

from .config import PROVIDER_CONFIGS, get_supported_providers
from .providers import (
    process_anthropic,
    process_deepseek,
    process_gemini,
    process_grok,
    process_kimi,
    process_minimax,
    process_openai,
    process_openrouter,
    process_qwen,
    process_stepfun,
    process_zhipu,
)

# Global provider function mapping for reuse across modules
PROVIDER_FUNCTIONS = {
    "openai": process_openai,
    "anthropic": process_anthropic,
    "deepseek": process_deepseek,
    "gemini": process_gemini,
    "qwen": process_qwen,
    "stepfun": process_stepfun,
    "zhipu": process_zhipu,
    "minimax": process_minimax,
    "grok": process_grok,
    "kimi": process_kimi,
    "openrouter": process_openrouter,
}

PROVIDER_MODEL_PREFIXES = {
    provider: config.model_prefixes
    for provider, config in PROVIDER_CONFIGS.items()
    if config.model_prefixes
}


def _validate_provider_function_registry() -> None:
    """Fail fast when configured providers and runtime implementations drift."""
    configured = set(get_supported_providers())
    implemented = set(PROVIDER_FUNCTIONS)
    if configured != implemented:
        missing = sorted(configured - implemented)
        unexpected = sorted(implemented - configured)
        raise RuntimeError(
            "Provider registry mismatch: "
            f"missing implementations={missing}, unexpected implementations={unexpected}"
        )


_validate_provider_function_registry()


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
    if not isinstance(model, str):
        raise ValueError(f"Model name must be a string, got {type(model).__name__}")

    model_normalized = model.strip()
    if not model_normalized:
        raise ValueError("Model name cannot be empty")

    model_lower = model_normalized.lower()

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
