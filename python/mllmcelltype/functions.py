from __future__ import annotations

from .providers import (
    process_anthropic,
    process_deepseek,
    process_gemini,
    process_grok,
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
    "openrouter": process_openrouter,
}

# Model prefix patterns for provider detection
# Each provider maps to a list of model name prefixes
# Order matters: more specific prefixes should come first
PROVIDER_MODEL_PREFIXES = {
    "openai": ["gpt-", "o1", "o3", "o4", "chatgpt-"],
    "anthropic": ["claude-"],
    "deepseek": ["deepseek-"],
    "gemini": ["gemini-"],
    "qwen": ["qwen", "qwq-"],
    "stepfun": ["step-"],
    "zhipu": ["glm-"],  # Supports glm-4.7, glm-4-plus, etc.
    "minimax": ["minimax-"],
    "grok": ["grok-"],
}


def get_provider(model: str) -> str:
    """Determine the provider based on the model name.

    Uses prefix matching for efficient provider detection.
    OpenRouter models are identified by the '/' character in the model name.

    Args:
        model: The model name (e.g., 'gpt-4o', 'claude-3-opus', 'anthropic/claude-3-opus')

    Returns:
        The provider name (e.g., 'openai', 'anthropic', 'openrouter')

    Raises:
        ValueError: If the provider cannot be determined from the model name
    """
    if not model:
        raise ValueError("Model name cannot be empty")

    model_lower = model.lower()

    # OpenRouter models contain '/' (e.g., 'anthropic/claude-3-opus')
    if "/" in model:
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
        f"Cannot determine provider for model: {model}. "
        f"Supported model prefixes: {', '.join(supported_prefixes)}"
    )
