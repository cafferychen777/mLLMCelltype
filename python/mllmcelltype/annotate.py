"""Main annotation module for LLMCellType."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from .config import get_default_model
from .execution import track_model_call
from .functions import PROVIDER_FUNCTIONS, validate_provider_model_match
from .logger import setup_logging, write_log
from .prompts import create_prompt
from .providers.common import normalize_response_lines
from .url_utils import resolve_provider_base_url
from .utils import (
    cluster_sort_key,
    create_cache_key,
    format_results,
    is_missing_value,
    load_api_key,
    load_from_cache,
    normalize_marker_genes_keys,
    parse_marker_genes,
    save_to_cache,
)
from .validation import normalize_text, validate_bool


def _resolve_provider(provider: str) -> tuple[str, Callable[..., Any]]:
    """Normalize provider name and resolve provider callable."""
    normalized_provider = normalize_text(provider, "provider", required=True).lower()
    provider_func = PROVIDER_FUNCTIONS.get(normalized_provider)
    if not provider_func:
        error_msg = f"Unknown provider: {normalized_provider}"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    return normalized_provider, provider_func


def _resolve_model(provider: str, model: str | None) -> str:
    """Resolve an explicit or provider-default model name."""
    resolved_model = normalize_text(model, "model")
    if not resolved_model:
        resolved_model = get_default_model(provider)
        write_log(f"Using default model for {provider}: {resolved_model}")
    validate_provider_model_match(provider, resolved_model, "model")
    return resolved_model


def _resolve_api_key(provider: str, api_key: str | None) -> str:
    """Resolve an API key only when an external provider call is required."""
    resolved_api_key = normalize_text(api_key, "api_key")
    if not resolved_api_key:
        resolved_api_key = normalize_text(load_api_key(provider), "api_key")
    if not resolved_api_key:
        error_msg = f"API key not found for provider: {provider}"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)
    return resolved_api_key


def _to_text_response(result: Any) -> str:
    """Validate and normalize provider response to text."""
    return "\n".join(normalize_response_lines(result, "model provider"))


def _normalize_annotation_response(
    result: Any,
    return_reasoning: bool = False,
) -> list[str] | dict[str, Any] | str:
    """Validate provider/cache payload before annotation parsing.

    When ``return_reasoning`` is True the original response text is preserved
    so that downstream JSON parsing sees intact punctuation (e.g. trailing
    commas inside JSON arrays/objects). Otherwise the response is normalized
    into non-empty lines as before.
    """
    if isinstance(result, dict):
        allowed_value_types = (str, dict) if return_reasoning else str
        for annotation in result.values():
            if not is_missing_value(annotation) and not isinstance(annotation, allowed_value_types):
                raise ValueError("Annotation response mappings must contain string values")
        return result
    if isinstance(result, str):
        if not result.strip():
            raise ValueError("Annotation response must contain non-empty text")
        return (
            result if return_reasoning else normalize_response_lines(result, "annotation provider")
        )
    if return_reasoning:
        raise ValueError("Reasoning response must be a string or dictionary")
    return normalize_response_lines(result, "annotation provider")


def _load_valid_cached_response(
    *,
    cache_key: str | None,
    cache_dir: str | None,
    normalizer: Callable[[Any], Any],
    response_kind: str,
) -> Any | None:
    """Load and validate cached data through one shared cache-read policy."""
    if cache_key is None:
        return None

    cached_response = load_from_cache(cache_key, cache_dir)
    if cached_response is None:
        return None

    try:
        return normalizer(cached_response)
    except ValueError as error:
        write_log(
            f"Ignoring invalid cached {response_kind} response: {error!s}",
            level="warning",
        )
        return None


def _split_analyzable_clusters(
    marker_genes: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Split marker genes into analyzable and empty-marker cluster IDs."""
    analyzable: dict[str, list[str]] = {}
    empty_clusters: list[str] = []
    for cluster_id, genes in marker_genes.items():
        if genes:
            analyzable[cluster_id] = genes
        else:
            empty_clusters.append(cluster_id)
    return analyzable, empty_clusters


def _merge_annotation_results(
    *,
    all_clusters: list[str],
    analyzed_results: dict[str, str] | dict[str, dict[str, str]],
    empty_clusters: list[str],
    return_reasoning: bool = False,
) -> dict[str, str] | dict[str, dict[str, str]]:
    """Merge model annotations with deterministic Unknown for empty-marker clusters."""
    empty_cluster_set = set(empty_clusters)

    if return_reasoning:
        unknown: dict[str, str] = {
            "cell_type": "Unknown",
            "marker_genes": "",
            "gene_expression": "",
        }
        return {
            cluster_id: (
                unknown
                if cluster_id in empty_cluster_set
                else analyzed_results.get(cluster_id, unknown)
            )
            for cluster_id in all_clusters
        }

    return {
        cluster_id: (
            "Unknown"
            if cluster_id in empty_cluster_set
            else analyzed_results.get(cluster_id, "Unknown")
        )
        for cluster_id in all_clusters
    }


@track_model_call
def annotate_clusters(
    marker_genes: dict[str, list[str]] | pd.DataFrame,
    species: str,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    tissue: str | None = None,
    additional_context: str | None = None,
    prompt_template: str | None = None,
    use_cache: bool = True,
    cache_dir: str | None = None,
    log_dir: str | None = None,
    log_level: str = "INFO",
    base_urls: str | dict[str, str] | None = None,
    return_reasoning: bool = False,
) -> dict[str, str] | dict[str, dict[str, str]]:
    """Annotate cell clusters using LLM.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes,
                     or DataFrame with 'cluster' and 'gene' columns
        species: Species name (e.g., 'human', 'mouse')
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-5.5', 'claude-sonnet-4-6')
        api_key: API key for the provider
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        log_dir: Directory to store log files
        log_level: Logging level
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs (e.g., {'openai': 'https://proxy.com/v1'})
        return_reasoning: If True, return a structured dict per cluster with
            'cell_type', 'marker_genes', and 'gene_expression' fields instead of
            plain labels.

    Returns:
        Dict[str, str]: Dictionary mapping cluster names to annotations when
            return_reasoning is False.
        Dict[str, Dict[str, str]]: Dictionary mapping cluster names to reasoning
            records when return_reasoning is True.

    """
    use_cache = validate_bool(use_cache, "use_cache")
    return_reasoning = validate_bool(return_reasoning, "return_reasoning")
    species = normalize_text(species, "species", required=True)

    # Setup logging
    setup_logging(log_dir=log_dir, log_level=log_level)

    # Normalize provider to lowercase early so all downstream code
    # (base_urls lookup, cache key, API key resolution) is consistent.
    provider, provider_func = _resolve_provider(provider)
    write_log(f"Starting annotation with provider: {provider}")

    # Parse marker genes if DataFrame
    if isinstance(marker_genes, pd.DataFrame):
        marker_genes = parse_marker_genes(marker_genes)
    else:
        marker_genes = normalize_marker_genes_keys(marker_genes)

    # Get clusters in natural numerical order (consistent with prompt)
    all_clusters = sorted(marker_genes.keys(), key=cluster_sort_key)
    write_log(f"Found {len(all_clusters)} clusters")

    if not all_clusters:
        write_log("No clusters provided; skipping annotation request", level="warning")
        return {}

    analyzable_marker_genes, empty_clusters = _split_analyzable_clusters(marker_genes)
    analyzable_clusters = sorted(analyzable_marker_genes.keys(), key=cluster_sort_key)

    if empty_clusters:
        write_log(
            f"Skipping {len(empty_clusters)} cluster(s) with empty marker genes: {', '.join(sorted(empty_clusters, key=cluster_sort_key))}",
            level="warning",
        )

    if not analyzable_clusters:
        write_log(
            "No clusters have non-empty marker genes; returning Unknown for all clusters",
            level="warning",
        )
        if return_reasoning:
            unknown = {"cell_type": "Unknown", "marker_genes": "", "gene_expression": ""}
            return {cluster_id: unknown for cluster_id in all_clusters}
        return {cluster_id: "Unknown" for cluster_id in all_clusters}

    model = _resolve_model(provider, model)

    # Create prompt
    prompt = create_prompt(
        marker_genes=analyzable_marker_genes,
        species=species,
        tissue=tissue,
        additional_context=additional_context,
        prompt_template=prompt_template,
        return_reasoning=return_reasoning,
    )

    # Resolve base URL (before cache check — base_url is part of the cache key)
    base_url = resolve_provider_base_url(provider, base_urls)

    # Check cache
    cache_key = create_cache_key(prompt, model, provider, base_url) if use_cache else None
    cached_results = _load_valid_cached_response(
        cache_key=cache_key,
        cache_dir=cache_dir,
        normalizer=lambda result: _normalize_annotation_response(
            result, return_reasoning=return_reasoning
        ),
        response_kind="annotation",
    )
    if cached_results is not None:
        write_log("Using cached results")
        analyzed_results = format_results(cached_results, analyzable_clusters, return_reasoning)
        return _merge_annotation_results(
            all_clusters=all_clusters,
            analyzed_results=analyzed_results,
            empty_clusters=empty_clusters,
            return_reasoning=return_reasoning,
        )

    api_key = _resolve_api_key(provider, api_key)

    # Process request
    try:
        write_log(f"Processing request with {provider} using model {model}")
        start_time = time.time()

        # Call provider function with base_url; preserve raw response text for
        # reasoning mode so JSON punctuation survives line normalization.
        results = _normalize_annotation_response(
            provider_func(
                prompt,
                model,
                api_key,
                base_url,
                normalize_response=not return_reasoning,
            ),
            return_reasoning=return_reasoning,
        )

        end_time = time.time()
        write_log(f"Request processed in {end_time - start_time:.2f} seconds")

        # Save to cache
        if cache_key is not None:
            save_to_cache(cache_key, results, cache_dir)

        # Format results
        analyzed_results = format_results(results, analyzable_clusters, return_reasoning)
        return _merge_annotation_results(
            all_clusters=all_clusters,
            analyzed_results=analyzed_results,
            empty_clusters=empty_clusters,
            return_reasoning=return_reasoning,
        )

    except Exception as e:
        error_msg = f"Error during annotation: {e!s}"
        write_log(error_msg, level="error")
        raise


@track_model_call
def get_model_response(
    prompt: str,
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    use_cache: bool = True,
    cache_dir: str | None = None,
    base_url: str | None = None,
) -> str:
    """Get response from a model for a given prompt.

    Args:
        prompt: The prompt to send to the model
        provider: The provider name (e.g., 'openai', 'anthropic')
        model: The model name. If None, uses the default model for the provider.
        api_key: The API key for the provider. If None, loads from environment.
        use_cache: Whether to use cache
        cache_dir: The cache directory
        base_url: Optional custom base URL

    Returns:
        str: The model response

    """

    use_cache = validate_bool(use_cache, "use_cache")
    prompt = normalize_text(prompt, "prompt", required=True)
    provider, provider_func = _resolve_provider(provider)
    model = _resolve_model(provider, model)
    base_url = resolve_provider_base_url(provider, base_url)

    # Check cache
    cache_key = create_cache_key(prompt, model, provider, base_url) if use_cache else None
    cached_text = _load_valid_cached_response(
        cache_key=cache_key,
        cache_dir=cache_dir,
        normalizer=_to_text_response,
        response_kind="model",
    )
    if cached_text is not None:
        write_log(f"Using cached result for {model}")
        return cached_text

    api_key = _resolve_api_key(provider, api_key)

    # Call provider function
    try:
        write_log(f"Requesting response from {provider} ({model})")
        result = provider_func(prompt, model, api_key, base_url)
        text_result = _to_text_response(result)

        # Save to cache
        if cache_key is not None:
            save_to_cache(cache_key, text_result, cache_dir)

        return text_result
    except Exception as e:
        error_msg = f"Error getting model response: {e!s}"
        write_log(error_msg, level="error")
        raise
