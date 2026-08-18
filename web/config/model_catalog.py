"""Curated model catalog exposed by the web application.

The catalog intentionally contains only current general-purpose text models that
work with each provider's existing chat endpoint. Provider documentation was last
reviewed on 2026-07-14.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


MODEL_CATALOG_UPDATED_AT = "2026-08-18"


@dataclass(frozen=True)
class ModelOption:
    """A provider model identifier and its user-facing label."""

    id: str
    name: str


@dataclass(frozen=True)
class ProviderModelCatalog:
    """Current selectable models and the preferred default for one provider."""

    name: str
    default: str
    models: tuple[ModelOption, ...]


MODEL_CATALOG: Mapping[str, ProviderModelCatalog] = MappingProxyType(
    {
        "openai": ProviderModelCatalog(
            name="OpenAI",
            default="gpt-5.6-sol",
            models=(
                ModelOption("gpt-5.6-sol", "GPT-5.6 Sol"),
                ModelOption("gpt-5.6-terra", "GPT-5.6 Terra"),
                ModelOption("gpt-5.6-luna", "GPT-5.6 Luna"),
            ),
        ),
        "anthropic": ProviderModelCatalog(
            name="Anthropic",
            default="claude-sonnet-5",
            models=(
                ModelOption("claude-fable-5", "Claude Fable 5"),
                ModelOption("claude-opus-4-8", "Claude Opus 4.8"),
                ModelOption("claude-sonnet-5", "Claude Sonnet 5"),
                ModelOption("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
            ),
        ),
        "gemini": ProviderModelCatalog(
            name="Google (Gemini)",
            default="gemini-3.1-pro-preview",
            models=(
                ModelOption("gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
                ModelOption("gemini-3.5-flash", "Gemini 3.5 Flash"),
                ModelOption("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite"),
            ),
        ),
        "grok": ProviderModelCatalog(
            name="xAI (Grok)",
            default="grok-4.5",
            models=(
                ModelOption("grok-4.5", "Grok 4.5"),
                ModelOption("grok-4.3", "Grok 4.3"),
            ),
        ),
        "deepseek": ProviderModelCatalog(
            name="DeepSeek",
            default="deepseek-v4-flash",
            models=(
                ModelOption("deepseek-v4-pro", "DeepSeek V4 Pro"),
                ModelOption("deepseek-v4-flash", "DeepSeek V4 Flash"),
            ),
        ),
        "kimi": ProviderModelCatalog(
            name="Moonshot AI (Kimi)",
            default="kimi-k2.6",
            models=(ModelOption("kimi-k2.6", "Kimi K2.6"),),
        ),
        "qwen": ProviderModelCatalog(
            name="Alibaba (Qwen)",
            default="qwen3.7-plus",
            models=(
                ModelOption("qwen3.7-max", "Qwen 3.7 Max"),
                ModelOption("qwen3.7-plus", "Qwen 3.7 Plus"),
                ModelOption("qwen3.6-flash", "Qwen 3.6 Flash"),
            ),
        ),
        "zhipu": ProviderModelCatalog(
            name="Zhipu (GLM)",
            default="glm-5.1",
            models=(
                ModelOption("glm-5.1", "GLM-5.1"),
                ModelOption("glm-5", "GLM-5"),
                ModelOption("glm-5-turbo", "GLM-5 Turbo"),
            ),
        ),
        "stepfun": ProviderModelCatalog(
            name="StepFun",
            default="step-3.7-flash",
            models=(ModelOption("step-3.7-flash", "Step 3.7 Flash"),),
        ),
        "minimax": ProviderModelCatalog(
            name="MiniMax",
            default="MiniMax-M2.7",
            models=(
                ModelOption("MiniMax-M2.7", "MiniMax M2.7"),
                ModelOption("MiniMax-M2.7-highspeed", "MiniMax M2.7 Highspeed"),
            ),
        ),
        "openrouter": ProviderModelCatalog(
            name="OpenRouter",
            default="openrouter/auto",
            models=(
                ModelOption("openrouter/auto", "OpenRouter Auto Router"),
                ModelOption("openrouter/free", "OpenRouter Free Models Router"),
            ),
        ),
    }
)


def _validate_catalog() -> None:
    """Fail fast if a catalog edit introduces an invalid or duplicate entry."""
    for provider, catalog in MODEL_CATALOG.items():
        model_ids = [model.id for model in catalog.models]
        if not model_ids:
            raise ValueError(f"Model catalog for {provider} cannot be empty")
        if not catalog.name.strip():
            raise ValueError(f"Provider name for {provider} cannot be empty")
        if catalog.default not in model_ids:
            raise ValueError(f"Default model for {provider} is not selectable")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"Model catalog for {provider} contains duplicate IDs")
        if any(
            not model.id.strip() or not model.name.strip() for model in catalog.models
        ):
            raise ValueError(f"Model catalog for {provider} contains a blank value")


def get_provider_defaults() -> dict[str, str]:
    """Return a fresh provider-to-default-model mapping."""
    return {provider: catalog.default for provider, catalog in MODEL_CATALOG.items()}


def get_provider_names() -> dict[str, str]:
    """Return a fresh provider-to-display-name mapping."""
    return {provider: catalog.name for provider, catalog in MODEL_CATALOG.items()}


def get_serialized_catalog() -> dict[str, list[dict[str, str]]]:
    """Return JSON-safe model options without exposing mutable module state."""
    return {
        provider: [{"id": model.id, "name": model.name} for model in catalog.models]
        for provider, catalog in MODEL_CATALOG.items()
    }


_validate_catalog()
