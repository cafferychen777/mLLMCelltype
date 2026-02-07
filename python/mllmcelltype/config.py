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
    """

    default_model: str
    api_key_env_var: str
    default_api_url: str


# All provider configurations - Single Source of Truth
# When updating default models, only modify this dictionary
PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        default_model="gpt-5.2",
        api_key_env_var="OPENAI_API_KEY",
        default_api_url="https://api.openai.com/v1/chat/completions",
    ),
    "anthropic": ProviderConfig(
        default_model="claude-opus-4-6-20260205",
        api_key_env_var="ANTHROPIC_API_KEY",
        default_api_url="https://api.anthropic.com/v1/messages",
    ),
    "deepseek": ProviderConfig(
        default_model="deepseek-chat",  # V3.2
        api_key_env_var="DEEPSEEK_API_KEY",
        default_api_url="https://api.deepseek.com/v1/chat/completions",
    ),
    "gemini": ProviderConfig(
        default_model="gemini-3-pro",
        api_key_env_var="GEMINI_API_KEY",
        default_api_url="https://generativelanguage.googleapis.com/v1beta/models",
    ),
    "qwen": ProviderConfig(
        default_model="qwen3-max",
        api_key_env_var="QWEN_API_KEY",
        default_api_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
    ),
    "stepfun": ProviderConfig(
        default_model="step-3",
        api_key_env_var="STEPFUN_API_KEY",
        default_api_url="https://api.stepfun.com/v1/chat/completions",
    ),
    "zhipu": ProviderConfig(
        default_model="glm-4-plus",  # Stable model (glm-4.7 has rate limits)
        api_key_env_var="ZHIPU_API_KEY",
        default_api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
    ),
    "minimax": ProviderConfig(
        default_model="minimax-m2.1",
        api_key_env_var="MINIMAX_API_KEY",
        default_api_url="https://api.minimaxi.chat/v1/text/chatcompletion_v2",
    ),
    "grok": ProviderConfig(
        default_model="grok-4",
        api_key_env_var="GROK_API_KEY",
        default_api_url="https://api.x.ai/v1/chat/completions",
    ),
    "openrouter": ProviderConfig(
        default_model="openai/gpt-5.2",
        api_key_env_var="OPENROUTER_API_KEY",
        default_api_url="https://openrouter.ai/api/v1/chat/completions",
    ),
}


@dataclass(frozen=True)
class ConsensusConfig:
    """Configuration for consensus checking behavior.

    This defines which models to use for LLM-assisted consensus checking
    and what to fall back to when the primary fails.

    Design rationale:
    - Primary uses Qwen: Good cost-performance ratio, fast response, sufficient reasoning
    - Fallback uses Anthropic Sonnet: High stability, reliable as safety net
    """

    # Primary provider for consensus checking (used for LLM double-check)
    primary_provider: str = "qwen"

    # Fallback when primary fails
    # Using Sonnet (not Opus) for fallback because:
    # 1. Fallback should be reliable and fast, not necessarily the most powerful
    # 2. Sonnet has excellent stability and is cost-effective for this role
    fallback_provider: str = "anthropic"
    fallback_model: str = "claude-sonnet-4-5-20250929"

    @property
    def primary_model(self) -> str:
        """Get primary model from provider config.

        This ensures consistency with get_default_model().
        """
        return get_default_model(self.primary_provider)


# Default consensus configuration instance
DEFAULT_CONSENSUS_CONFIG = ConsensusConfig()

# Default fallback values when parsing fails or result is inconclusive
# These conservative values ensure discussion will be triggered
DEFAULT_FALLBACK_CONSENSUS_PROPORTION = 0.25
DEFAULT_FALLBACK_ENTROPY = 2.0


def get_default_model(provider: str) -> str:
    """Get default model for a provider.

    This is the ONLY function to get default models.
    All other code should call this function instead of hardcoding model names.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic', 'qwen')

    Returns:
        Default model name for the provider, or 'unknown' if provider not found
    """
    config = PROVIDER_CONFIGS.get(provider.lower())
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
    config = PROVIDER_CONFIGS.get(provider.lower())
    if config:
        return config.api_key_env_var
    # Fallback pattern for unknown providers
    return f"{provider.upper()}_API_KEY"


def get_default_api_url(provider: str) -> str:
    """Get default API URL for a provider.

    Args:
        provider: Provider name

    Returns:
        Default API URL, or empty string if provider not found
    """
    config = PROVIDER_CONFIGS.get(provider.lower())
    if config:
        return config.default_api_url
    return ""


def get_supported_providers() -> list[str]:
    """Get list of all supported provider names.

    Returns:
        List of provider names
    """
    return list(PROVIDER_CONFIGS.keys())
