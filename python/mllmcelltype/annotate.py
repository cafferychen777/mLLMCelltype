"""Main annotation module for LLMCellType."""

from __future__ import annotations

import time
from typing import Optional, Union

import pandas as pd

from .functions import PROVIDER_FUNCTIONS
from .logger import setup_logging, write_log
from .prompts import create_prompt
from .url_utils import resolve_provider_base_url
from .utils import (
    create_cache_key,
    format_results,
    load_api_key,
    load_from_cache,
    parse_marker_genes,
    save_to_cache,
)


def annotate_clusters(
    marker_genes: Union[dict[str, list[str]], pd.DataFrame],
    species: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    base_urls: Optional[Union[str, dict[str, str]]] = None,
) -> dict[str, str]:
    """Annotate cell clusters using LLM.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes,
                     or DataFrame with 'cluster' and 'gene' columns
        species: Species name (e.g., 'human', 'mouse')
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-5', 'claude-sonnet-4-5-20250929')
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

    Returns:
        Dict[str, str]: Dictionary mapping cluster names to annotations

    """
    # Setup logging
    setup_logging(log_dir=log_dir, log_level=log_level)
    write_log(f"Starting annotation with provider: {provider}")

    # Parse marker genes if DataFrame
    if isinstance(marker_genes, pd.DataFrame):
        marker_genes = parse_marker_genes(marker_genes)

    # Get clusters
    clusters = list(marker_genes.keys())
    write_log(f"Found {len(clusters)} clusters")

    # Set default model based on provider
    if not model:
        model = get_default_model(provider)
        write_log(f"Using default model for {provider}: {model}")

    # Get API key if not provided
    if not api_key:
        api_key = load_api_key(provider)
        if not api_key:
            error_msg = f"API key not found for provider: {provider}"
            write_log(error_msg, level="error")
            raise ValueError(error_msg)

    # Create prompt
    prompt = create_prompt(
        marker_genes=marker_genes,
        species=species,
        tissue=tissue,
        additional_context=additional_context,
        prompt_template=prompt_template,
    )

    # Check cache
    if use_cache:
        cache_key = create_cache_key(prompt, model, provider)
        cached_results = load_from_cache(cache_key, cache_dir)
        if cached_results:
            write_log("Using cached results")
            return format_results(cached_results, clusters)

    # Resolve base URL
    base_url = resolve_provider_base_url(provider, base_urls)

    # Get provider function
    provider_func = PROVIDER_FUNCTIONS.get(provider.lower())
    if not provider_func:
        error_msg = f"Unknown provider: {provider}"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    # Process request
    try:
        write_log(f"Processing request with {provider} using model {model}")
        start_time = time.time()

        # Call provider function with base_url
        results = provider_func(prompt, model, api_key, base_url)

        end_time = time.time()
        write_log(f"Request processed in {end_time - start_time:.2f} seconds")

        # Save to cache
        if use_cache:
            save_to_cache(cache_key, results, cache_dir)

        # Format results
        return format_results(results, clusters)

    except Exception as e:
        error_msg = f"Error during annotation: {str(e)}"
        write_log(error_msg, level="error")
        raise


def get_default_model(provider: str) -> str:
    """Get default model for a provider.

    Args:
        provider: Provider name

    Returns:
        str: Default model name

    """
    default_models = {
        "openai": "gpt-5.2",
        "anthropic": "claude-opus-4-5-20251101",
        "deepseek": "deepseek-chat",  # V3.2
        "gemini": "gemini-3-pro",
        "qwen": "qwen3-max",
        "stepfun": "step-3",
        "zhipu": "glm-4-plus",  # Stable model (glm-4.7 has rate limits)
        "minimax": "MiniMax-Text-01",
        "grok": "grok-3",
        "openrouter": "openai/gpt-5.2",
    }

    return default_models.get(provider.lower(), "unknown")


def get_model_response(
    prompt: str,
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    base_url: Optional[str] = None,
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

    # Check if provider is valid
    if not provider:
        raise ValueError("Provider name is required")

    # Set default model if not provided
    if not model:
        model = get_default_model(provider)
        write_log(f"Using default model for {provider}: {model}")

    # Get API key if not provided
    if not api_key:
        api_key = load_api_key(provider)
        if not api_key:
            error_msg = f"API key not found for provider: {provider}"
            write_log(error_msg, level="error")
            raise ValueError(error_msg)

    # Check cache
    if use_cache:
        cache_key = create_cache_key(prompt, model, provider)
        cached_result = load_from_cache(cache_key, cache_dir)
        if cached_result:
            write_log(f"Using cached result for {model}")
            if isinstance(cached_result, list):
                return "\n".join(cached_result)
            return cached_result

    # Get provider function
    provider_func = PROVIDER_FUNCTIONS.get(provider.lower())
    if not provider_func:
        error_msg = f"Unknown provider: {provider}"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    # Call provider function
    try:
        write_log(f"Requesting response from {provider} ({model})")
        result = provider_func(prompt, model, api_key, base_url)

        # Save to cache
        if use_cache:
            save_to_cache(cache_key, result, cache_dir)

        # Convert list to string if needed
        if isinstance(result, list):
            return "\n".join(result)

        return result
    except Exception as e:
        error_msg = f"Error getting model response: {str(e)}"
        write_log(error_msg, level="error")
        raise
