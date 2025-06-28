from __future__ import annotations

from typing import Literal

from .logger import write_log
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
from .utils import clean_annotation

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

# Define supported models as literals for better type checking
ModelType = Literal[
    # OpenAI models
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4.1",
    "o1",
    "o1-mini",
    "o1-pro",
    "o4-mini",
    # Anthropic models
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-reasoner",
    # Gemini models
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    # Qwen models
    "qwen-max-2025-01-25",
    "qwen3-72b",
    "qwen-plus",
    # StepFun models
    "step-2-16k",
    "step-2-mini",
    "step-1-8k",
    # Zhipu models
    "glm-4-plus",
    "glm-3-turbo",
    # MiniMax models
    "minimax-text-01",
    # Grok models
    "grok-3-latest",
    "grok-3",
]


def get_provider(model: str) -> str:
    """Determine the provider based on the model name."""
    # Special case for OpenRouter models which may contain '/' in the model name
    if isinstance(model, str) and "/" in model:
        # OpenRouter models are in the format 'provider/model'
        # e.g., 'anthropic/claude-3-opus'
        return "openrouter"

    # Common model prefixes for each provider (based on R package and API verification)
    providers = {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o1-preview",
            "o1-pro",
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-4.5-preview",
            "gpt-4",
            "chatgpt-4o-latest",
        ],
        "anthropic": [
            "claude-opus-4-20250514",
            "claude-opus-4",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus",
        ],
        "deepseek": ["deepseek-chat", "deepseek-r1", "deepseek-r1-zero", "deepseek-reasoner"],
        "gemini": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-pro-preview-06-05",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-lite-001",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-002",
            "gemini-1.5-pro",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash-8b-001",
        ],
        "qwen": [
            "qwen-max",
            "qwen-max-2025-01-25",
            "qwen-plus",
            "qwen-turbo",
            "qwen-long",
            "qwen2.5-72b-instruct",
            "qwen2.5-32b-instruct",
            "qwen2.5-14b-instruct",
            "qwen2.5-7b-instruct",
            "qwen2.5-3b-instruct",
            "qwen2.5-1.5b-instruct",
            "qwen2.5-0.5b-instruct",
            "qwen3-72b",
            "qwq-32b-preview",
        ],
        "stepfun": ["step-2-mini", "step-2-16k", "step-1-8k"],
        "zhipu": [
            "glm-4-plus",
            "glm-4",
            "glm-4-0520",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-flash",
            "glm-4-flashx",
            "glm-4v",
            "glm-4v-plus",
            "glm-3-turbo",
        ],
        "minimax": ["minimax-text-01", "minimax-01", "minimax-m1"],
        "grok": [
            "grok-3",
            "grok-3-fast",
            "grok-3-mini",
            "grok-3-mini-fast",
            "grok-2-1212",
            "grok-3-beta",
            "grok-3-mini-beta",
            "grok-beta",
        ],
        "openrouter": [
            # Any model with "/" will be detected as OpenRouter
            # These are some popular OpenRouter models for reference
            "anthropic/claude-opus-4",
            "anthropic/claude-sonnet-4",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "openai/gpt-4.1",
            "openai/o1",
            "openai/o3",
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-4-maverick",
            "google/gemini-2.5-pro",
            "mistralai/mistral-large",
            "deepseek/deepseek-r1",
            "microsoft/phi-4",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "x-ai/grok-3-beta",
            "qwen/qwq-32b",
            "perplexity/sonar-pro",
        ],
    }

    # Check for model name in each provider's list
    for provider, models in providers.items():
        for supported_model in models:
            if model.lower() == supported_model.lower() or model.lower().startswith(
                supported_model.lower()
            ):
                return provider

    # Check for provider name in the model string (fallback)
    for provider in providers:
        if provider.lower() in model.lower():
            return provider

    # List all supported models for the error message
    all_supported = []
    for _provider, models in providers.items():
        all_supported.extend(models)

    write_log(
        f"WARNING: Unsupported model: {model}. Using provider name from model string.",
        "warning",
    )

    # Try to extract provider name from the model string
    for known_provider in [
        "openai",
        "anthropic",
        "claude",
        "gpt",
        "deepseek",
        "gemini",
        "google",
        "qwen",
        "alibaba",
        "step",
        "glm",
        "zhipu",
        "minimax",
        "grok",
        "xai",
    ]:
        if known_provider in model.lower():
            if known_provider == "gpt":
                return "openai"
            if known_provider == "claude":
                return "anthropic"
            if known_provider == "google":
                return "gemini"
            if known_provider == "alibaba":
                return "qwen"
            if known_provider == "glm":
                return "zhipu"
            if known_provider == "xai":
                return "grok"
            return known_provider

    # If we still can't determine the provider, raise an error
    raise ValueError(
        f"Unsupported model: {model}. Supported models are: {', '.join(all_supported)}"
    )


def select_best_prediction(predictions: list[dict[str, str]]) -> dict[str, str]:
    """Select the best prediction from multiple models.

    Args:
        predictions: List of dictionaries mapping cluster IDs to cell type annotations

    Returns:
        dict[str, str]: Dictionary mapping cluster IDs to best predictions

    """
    if not predictions:
        return {}

    # Get all cluster IDs
    all_clusters = set()
    for prediction in predictions:
        all_clusters.update(prediction.keys())

    # For each cluster, select the most specific prediction
    best_predictions = {}
    for cluster in all_clusters:
        cluster_predictions = [pred.get(cluster, "") for pred in predictions if cluster in pred]

        # Filter out empty predictions
        cluster_predictions = [pred for pred in cluster_predictions if pred]

        if not cluster_predictions:
            best_predictions[cluster] = "Unknown"
            continue

        # Select the longest prediction (assuming it's more specific)
        # This is a simple heuristic and could be improved
        best_pred = max(cluster_predictions, key=len)
        best_predictions[cluster] = best_pred

    return best_predictions


def identify_controversial_clusters(
    annotations: dict[str, dict[str, str]], threshold: float = 0.6
) -> list[str]:
    """Identify clusters with inconsistent annotations across models.

    Args:
        annotations: Dictionary mapping model names to dictionaries of cluster annotations
        threshold: Agreement threshold below which a cluster is considered controversial

    Returns:
        list[str]: List of controversial cluster IDs

    """
    if not annotations or len(annotations) < 2:
        return []

    # Get all clusters
    all_clusters = set()
    for model_results in annotations.values():
        all_clusters.update(model_results.keys())

    controversial = []

    # Check each cluster for agreement level
    for cluster in all_clusters:
        # Get all annotations for this cluster
        cluster_annotations = []
        for _model, results in annotations.items():
            if cluster in results:
                annotation = clean_annotation(results[cluster])
                if annotation:
                    cluster_annotations.append(annotation)

        # Count occurrences
        counts = {}
        for anno in cluster_annotations:
            counts[anno] = counts.get(anno, 0) + 1

        # Find most common annotation and its frequency
        if counts:
            most_common = max(counts.items(), key=lambda x: x[1])
            most_common_count = most_common[1]
            agreement = most_common_count / len(cluster_annotations) if cluster_annotations else 0

            # Mark as controversial if agreement is below threshold
            if agreement < threshold:
                controversial.append(cluster)

    return controversial
