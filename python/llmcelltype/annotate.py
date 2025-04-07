"""
Main annotation module for LLMCellType.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import time

from .providers import (
    process_openai,
    process_anthropic,
    process_deepseek,
    process_gemini,
    process_qwen,
    process_stepfun,
    process_zhipu,
    process_minimax
)
from .prompts import create_prompt, create_batch_prompt
from .utils import (
    load_api_key,
    create_cache_key,
    save_to_cache,
    load_from_cache,
    parse_marker_genes,
    format_results
)
from .logger import write_log, setup_logging

# Provider function mapping
PROVIDER_FUNCTIONS = {
    'openai': process_openai,
    'anthropic': process_anthropic,
    'deepseek': process_deepseek,
    'gemini': process_gemini,
    'qwen': process_qwen,
    'stepfun': process_stepfun,
    'zhipu': process_zhipu,
    'minimax': process_minimax
}

def annotate_clusters(
    marker_genes: Union[Dict[str, List[str]], pd.DataFrame],
    species: str,
    provider: str = 'openai',
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_level: str = 'INFO'
) -> Dict[str, str]:
    """
    Annotate cell clusters using LLM.
    
    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes,
                     or DataFrame with 'cluster' and 'gene' columns
        species: Species name (e.g., 'human', 'mouse')
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-4o', 'claude-3-opus-20240229')
        api_key: API key for the provider
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        log_dir: Directory to store log files
        log_level: Logging level
        
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
            write_log(f"ERROR: {error_msg}", level='error')
            raise ValueError(error_msg)
    
    # Create prompt
    prompt = create_prompt(
        marker_genes=marker_genes,
        species=species,
        tissue=tissue,
        additional_context=additional_context,
        prompt_template=prompt_template
    )
    
    # Check cache
    if use_cache:
        cache_key = create_cache_key(prompt, model, provider)
        cached_results = load_from_cache(cache_key, cache_dir)
        if cached_results:
            write_log("Using cached results")
            return format_results(cached_results, clusters)
    
    # Get provider function
    provider_func = PROVIDER_FUNCTIONS.get(provider.lower())
    if not provider_func:
        error_msg = f"Unknown provider: {provider}"
        write_log(f"ERROR: {error_msg}", level='error')
        raise ValueError(error_msg)
    
    # Process request
    try:
        write_log(f"Processing request with {provider} using model {model}")
        start_time = time.time()
        
        # Call provider function
        results = provider_func(prompt, model, api_key)
        
        end_time = time.time()
        write_log(f"Request processed in {end_time - start_time:.2f} seconds")
        
        # Save to cache
        if use_cache:
            save_to_cache(cache_key, results, cache_dir)
        
        # Format results
        return format_results(results, clusters)
    
    except Exception as e:
        error_msg = f"Error during annotation: {str(e)}"
        write_log(f"ERROR: {error_msg}", level='error')
        raise

def batch_annotate_clusters(
    marker_genes_list: List[Union[Dict[str, List[str]], pd.DataFrame]],
    species: str,
    provider: str = 'openai',
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    tissue: Optional[Union[str, List[str]]] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_level: str = 'INFO'
) -> List[Dict[str, str]]:
    """
    Batch annotate multiple sets of cell clusters using LLM.
    
    Args:
        marker_genes_list: List of dictionaries mapping cluster names to lists of marker genes,
                          or list of DataFrames with 'cluster' and 'gene' columns
        species: Species name (e.g., 'human', 'mouse')
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-4o', 'claude-3-opus-20240229')
        api_key: API key for the provider
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        log_dir: Directory to store log files
        log_level: Logging level
        
    Returns:
        List[Dict[str, str]]: List of dictionaries mapping cluster names to annotations
    """
    # Setup logging
    setup_logging(log_dir=log_dir, log_level=log_level)
    write_log(f"Starting batch annotation with provider: {provider}")
    
    # Parse marker genes if DataFrames
    parsed_marker_genes_list = []
    for marker_genes in marker_genes_list:
        if isinstance(marker_genes, pd.DataFrame):
            parsed_marker_genes_list.append(parse_marker_genes(marker_genes))
        else:
            parsed_marker_genes_list.append(marker_genes)
    
    # Get clusters for each set
    clusters_list = [list(marker_genes.keys()) for marker_genes in parsed_marker_genes_list]
    write_log(f"Found {len(clusters_list)} sets of clusters")
    
    # Set default model based on provider
    if not model:
        model = get_default_model(provider)
        write_log(f"Using default model for {provider}: {model}")
    
    # Get API key if not provided
    if not api_key:
        api_key = load_api_key(provider)
        if not api_key:
            error_msg = f"API key not found for provider: {provider}"
            write_log(f"ERROR: {error_msg}", level='error')
            raise ValueError(error_msg)
    
    # Create batch prompt
    # If tissue is a list, use the first one for the batch prompt
    # Individual tissues will be used when processing each dataset separately
    batch_tissue = tissue[0] if isinstance(tissue, list) and len(tissue) > 0 else tissue
    
    prompt = create_batch_prompt(
        marker_genes_list=parsed_marker_genes_list,
        species=species,
        tissue=batch_tissue,
        additional_context=additional_context,
        prompt_template=prompt_template
    )
    
    # Check cache
    if use_cache:
        cache_key = create_cache_key(prompt, model, provider)
        cached_results = load_from_cache(cache_key, cache_dir)
        if cached_results:
            write_log("Using cached results")
            # Parse cached results into sets
            result_sets = []
            start_idx = 0
            for clusters in clusters_list:
                end_idx = start_idx + len(clusters)
                set_results = cached_results[start_idx:end_idx]
                result_sets.append(format_results(set_results, clusters))
                start_idx = end_idx
            return result_sets
    
    # Get provider function
    provider_func = PROVIDER_FUNCTIONS.get(provider.lower())
    if not provider_func:
        error_msg = f"Unknown provider: {provider}"
        write_log(f"ERROR: {error_msg}", level='error')
        raise ValueError(error_msg)
    
    # Process request
    try:
        write_log(f"Processing batch request with {provider} using model {model}")
        start_time = time.time()
        
        # Call provider function
        results = provider_func(prompt, model, api_key)
        
        end_time = time.time()
        write_log(f"Batch request processed in {end_time - start_time:.2f} seconds")
        
        # Save to cache
        if use_cache:
            save_to_cache(cache_key, results, cache_dir)
        
        # Parse results into sets
        # The LLM response format is typically:
        # Set 1:
        # Cluster 1: T cells
        # Cluster 2: B cells
        # ...
        # 
        # Set 2:
        # ...
        
        # First, filter out empty lines and parse the results
        filtered_results = [line for line in results if line.strip()]
        
        # Initialize result sets
        result_sets = []
        current_set = None
        current_set_results = {}
        
        for line in filtered_results:
            line = line.strip()
            
            # Check if this is a set header
            if line.startswith('Set '):
                # If we have a current set, add it to results
                if current_set is not None and current_set_results:
                    result_sets.append(current_set_results)
                    current_set_results = {}
                
                # Extract set number
                try:
                    current_set = int(line.split()[1].rstrip(':')) - 1
                except (IndexError, ValueError):
                    current_set = len(result_sets)
                
            # Check if this is a cluster annotation
            elif line.startswith('Cluster '):
                try:
                    # Parse "Cluster X: Annotation"
                    parts = line.split(':', 1)
                    cluster_num = parts[0].split()[1]
                    annotation = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Add to current set results
                    if current_set is not None:
                        current_set_results[cluster_num] = annotation
                except (IndexError, ValueError):
                    write_log(f"Warning: Could not parse line: {line}", level='warning')
        
        # Add the last set if it exists
        if current_set is not None and current_set_results:
            result_sets.append(current_set_results)
        
        # Try to parse JSON format if present
        if not result_sets:
            try:
                # Join all results and try to parse as JSON
                full_text = "\n".join(filtered_results)
                
                # Try to find JSON in the text
                import re
                import json
                
                # Extract JSON content if it's wrapped in ```json and ``` markers
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', full_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no code blocks, try to find JSON object directly
                    json_match = re.search(r'(\{[\s\S]*\})', full_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = None
                
                if json_str:
                    try:
                        data = json.loads(json_str)
                        
                        # Check if it's a batch response with sets
                        if "sets" in data and isinstance(data["sets"], list):
                            # Parse each set
                            for set_data in data["sets"]:
                                set_results = {}
                                if "clusters" in set_data and isinstance(set_data["clusters"], list):
                                    for cluster in set_data["clusters"]:
                                        if "id" in cluster and "cell_type" in cluster:
                                            set_results[str(cluster["id"])] = cluster["cell_type"]
                                    result_sets.append(set_results)
                        # Check if it's a batch response with annotations array
                        elif "annotations" in data and isinstance(data["annotations"], list):
                            # Group by set if specified
                            set_groups = {}
                            for annotation in data["annotations"]:
                                if "set" in annotation and "cluster" in annotation and "cell_type" in annotation:
                                    set_id = annotation["set"]
                                    if set_id not in set_groups:
                                        set_groups[set_id] = {}
                                    set_groups[set_id][str(annotation["cluster"])] = annotation["cell_type"]
                            
                            # Add all sets in order
                            for set_id in sorted(set_groups.keys()):
                                result_sets.append(set_groups[set_id])
                    except json.JSONDecodeError:
                        write_log(f"Warning: Failed to parse JSON response", level='warning')
            except Exception as e:
                write_log(f"Warning: Error while trying to parse JSON: {str(e)}", level='warning')
        
        # If we still didn't get any properly formatted results, fall back to the old method
        if not result_sets:
            write_log("Warning: Could not parse batch results, falling back to default parsing", level='warning')
            result_sets = []
            for i, clusters in enumerate(clusters_list):
                result_sets.append({str(cluster): f"Set {i+1} results not properly parsed" for cluster in clusters})
        
        return result_sets
    
    except Exception as e:
        error_msg = f"Error during batch annotation: {str(e)}"
        write_log(f"ERROR: {error_msg}", level='error')
        raise

def get_default_model(provider: str) -> str:
    """
    Get default model for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        str: Default model name
    """
    default_models = {
        'openai': 'gpt-4o',
        'anthropic': 'claude-3-opus-20240229',
        'deepseek': 'deepseek-chat',
        'gemini': 'gemini-2.0-pro',
        'qwen': 'qwen-max-2025-01-25',
        'stepfun': 'llama3-70b-8192',
        'zhipu': 'glm-4',
        'minimax': 'abab6-chat'
    }
    
    return default_models.get(provider.lower(), 'unknown')

def get_model_response(
    prompt: str,
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None
) -> str:
    """
    Get response from a model for a given prompt.
    
    Args:
        prompt: The prompt to send to the model
        provider: The provider name (e.g., 'openai', 'anthropic')
        model: The model name. If None, uses the default model for the provider.
        api_key: The API key for the provider. If None, loads from environment.
        use_cache: Whether to use cache
        cache_dir: The cache directory
        
    Returns:
        str: The model response
    """
    from .functions import get_provider
    
    # Check if provider is valid
    if not provider:
        raise ValueError("Provider name is required")
    
    # Set default model if not provided
    if not model:
        model = get_default_model(provider)
        write_log(f"Using default model for {provider}: {model}")
    
    # Get API key if not provided
    if not api_key:
        from .utils import load_api_key
        api_key = load_api_key(provider)
        if not api_key:
            error_msg = f"API key not found for provider: {provider}"
            write_log(f"ERROR: {error_msg}", level='error')
            raise ValueError(error_msg)
    
    # Check cache
    if use_cache:
        from .utils import create_cache_key, load_from_cache
        cache_key = create_cache_key(prompt, model, provider)
        cached_result = load_from_cache(cache_key, cache_dir)
        if cached_result:
            write_log(f"Using cached result for {model}")
            if isinstance(cached_result, list):
                return "\n".join(cached_result)
            return cached_result
    
    # Get provider function
    provider_funcs = {
        'openai': PROVIDER_FUNCTIONS['openai'],
        'anthropic': PROVIDER_FUNCTIONS['anthropic'],
        'deepseek': PROVIDER_FUNCTIONS['deepseek'],
        'gemini': PROVIDER_FUNCTIONS['gemini'],
        'qwen': PROVIDER_FUNCTIONS['qwen'],
        'stepfun': PROVIDER_FUNCTIONS['stepfun'],
        'zhipu': PROVIDER_FUNCTIONS['zhipu'],
        'minimax': PROVIDER_FUNCTIONS['minimax']
    }
    
    provider_func = provider_funcs.get(provider.lower())
    if not provider_func:
        error_msg = f"Unknown provider: {provider}"
        write_log(f"ERROR: {error_msg}", level='error')
        raise ValueError(error_msg)
    
    # Call provider function
    try:
        write_log(f"Requesting response from {provider} ({model})")
        result = provider_func(prompt, model, api_key)
        
        # Save to cache
        if use_cache:
            from .utils import save_to_cache
            save_to_cache(cache_key, result, cache_dir)
        
        # Convert list to string if needed
        if isinstance(result, list):
            return "\n".join(result)
        
        return result
    except Exception as e:
        error_msg = f"Error getting model response: {str(e)}"
        write_log(f"ERROR: {error_msg}", level='error')
        raise
