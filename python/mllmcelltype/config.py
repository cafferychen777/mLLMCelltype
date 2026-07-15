"""Centralized configuration for model defaults and provider settings.

This module serves as the Single Source of Truth for all provider configurations.
All default models, API URLs, and environment variable names should be defined here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a single LLM provider.

    Attributes:
        default_model: The default model to use when none is specified
        api_key_env_var: Environment variable name for the API key
        default_api_url: Default API endpoint URL
        model_prefixes: Prefixes used to infer this provider from a model name
    """

    default_model: str
    api_key_env_var: str
    default_api_url: str
    model_prefixes: tuple[str, ...] = ()


# All provider configurations - Single Source of Truth
# When updating default models, only modify this dictionary
PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        default_model="gpt-5.5",
        api_key_env_var="OPENAI_API_KEY",
        default_api_url="https://api.openai.com/v1/chat/completions",
        model_prefixes=("gpt-", "o1", "o3", "o4", "chatgpt-", "codex-"),
    ),
    "anthropic": ProviderConfig(
        default_model="claude-opus-4-7",
        api_key_env_var="ANTHROPIC_API_KEY",
        default_api_url="https://api.anthropic.com/v1/messages",
        model_prefixes=("claude-",),
    ),
    "deepseek": ProviderConfig(
        default_model="deepseek-v4-flash",
        api_key_env_var="DEEPSEEK_API_KEY",
        default_api_url="https://api.deepseek.com/v1/chat/completions",
        model_prefixes=("deepseek-",),
    ),
    "gemini": ProviderConfig(
        default_model="gemini-3.1-pro-preview",
        api_key_env_var="GEMINI_API_KEY",
        default_api_url="https://generativelanguage.googleapis.com/v1beta/models",
        model_prefixes=("gemini-",),
    ),
    "qwen": ProviderConfig(
        default_model="qwen3.6-plus",
        api_key_env_var="QWEN_API_KEY",
        default_api_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
        model_prefixes=("qwen", "qwq-"),
    ),
    "stepfun": ProviderConfig(
        default_model="step-3.5-flash",
        api_key_env_var="STEPFUN_API_KEY",
        default_api_url="https://api.stepfun.com/v1/chat/completions",
        model_prefixes=("step-",),
    ),
    "zhipu": ProviderConfig(
        default_model="glm-5.1",
        api_key_env_var="ZHIPU_API_KEY",
        default_api_url="https://api.z.ai/api/paas/v4/chat/completions",
        model_prefixes=("glm-", "chatglm"),
    ),
    "minimax": ProviderConfig(
        default_model="MiniMax-M2.7",
        api_key_env_var="MINIMAX_API_KEY",
        default_api_url="https://api.minimax.io/v1/chat/completions",
        model_prefixes=("minimax-",),
    ),
    "grok": ProviderConfig(
        default_model="grok-4.3",
        api_key_env_var="GROK_API_KEY",
        default_api_url="https://api.x.ai/v1/chat/completions",
        model_prefixes=("grok-",),
    ),
    "kimi": ProviderConfig(
        default_model="kimi-k2.7",
        api_key_env_var="KIMI_API_KEY",
        default_api_url="https://api.kimi.com/coding/v1/messages",
        model_prefixes=("kimi-", "moonshot"),
    ),
    "openrouter": ProviderConfig(
        default_model="openai/gpt-5.5",
        api_key_env_var="OPENROUTER_API_KEY",
        default_api_url="https://openrouter.ai/api/v1/chat/completions",
    ),
}

# Default fallback values when parsing fails or result is inconclusive
# These conservative values ensure discussion will be triggered
DEFAULT_FALLBACK_CONSENSUS_PROPORTION = 0.25
DEFAULT_FALLBACK_ENTROPY = 2.0


def _normalize_provider_name(provider: object) -> str:
    """Normalize provider name for case-insensitive lookups."""
    if not isinstance(provider, str):
        return ""
    return provider.strip().lower()


def get_default_model(provider: str) -> str:
    """Get default model for a provider.

    This is the ONLY function to get default models.
    All other code should call this function instead of hardcoding model names.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic', 'qwen')

    Returns:
        Default model name for the provider, or 'unknown' if provider not found
    """
    provider_name = _normalize_provider_name(provider)
    config = PROVIDER_CONFIGS.get(provider_name)
    if config:
        return config.default_model
    return "unknown"


def get_api_key_env_var(provider: str) -> str:
    """Get environment variable name for a provider's API key.

    Args:
        provider: Provider name

    Returns:
        Environment variable name (e.g., 'OPENAI_API_KEY')
    """
    provider_name = _normalize_provider_name(provider)
    config = PROVIDER_CONFIGS.get(provider_name)
    if config:
        return config.api_key_env_var
    # Fallback pattern for unknown providers
    fallback = str(provider).strip().upper() if provider is not None else "UNKNOWN"
    return f"{fallback}_API_KEY"


def get_default_api_url(provider: str) -> str:
    """Get default API URL for a provider.

    Args:
        provider: Provider name

    Returns:
        Default API URL, or empty string if provider not found
    """
    provider_name = _normalize_provider_name(provider)
    config = PROVIDER_CONFIGS.get(provider_name)
    if config:
        return config.default_api_url
    return ""


def get_supported_providers() -> list[str]:
    """Get list of all supported provider names.

    Returns:
        List of provider names
    """
    return list(PROVIDER_CONFIGS.keys())
