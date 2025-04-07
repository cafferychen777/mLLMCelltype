import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Literal, Any, Optional, Tuple
import openai
import json
import requests
import time
import logging
from dataclasses import dataclass

from .logger import write_log
from .utils import clean_annotation

# Define supported models as literals for better type checking
ModelType = Literal[
    # OpenAI models
    'gpt-4o', 'gpt-4-turbo', 'o1', 'o1-mini',
    # Anthropic models
    'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus',
    # DeepSeek models
    'deepseek-chat', 'deepseek-reasoner',
    # Gemini models
    'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash',
    # Qwen models
    'qwen-max-2025-01-25', 'qwen-plus',
    # StepFun models
    'step-2-16k', 'step-2-mini', 'step-1-8k',
    # Zhipu models
    'glm-4-plus', 'glm-3-turbo',
    # MiniMax models
    'minimax-text-01'
]

@dataclass
class LLMResponse:
    """Class for storing LLM response data"""
    cell_types: List[str]
    prompt: str
    raw_response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def get_provider(model: str) -> str:
    """
    Determine the provider based on the model name.
    
    Args:
        model: Name of the model
    
    Returns:
        str: Name of the provider ('openai', 'anthropic', 'deepseek', 'gemini', 'qwen', etc.)
    
    Raises:
        ValueError: If the model is not supported
    """
    # Define supported models
    providers = {
        'openai': ['gpt-4o', 'gpt-4-turbo', 'o1', 'o1-mini'],
        'anthropic': ['claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 
                     'claude-3-5-haiku-latest', 'claude-3-opus'],
        'deepseek': ['deepseek-chat', 'deepseek-reasoner'],
        'gemini': ['gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
        'qwen': ['qwen-max-2025-01-25', 'qwen-plus'],
        'stepfun': ['step-2-16k', 'step-2-mini', 'step-1-8k'],
        'zhipu': ['glm-4-plus', 'glm-3-turbo'],
        'minimax': ['minimax-text-01']
    }
    
    # Check for model name in each provider's list
    for provider, models in providers.items():
        for supported_model in models:
            if model.lower() == supported_model.lower() or model.lower().startswith(supported_model.lower()):
                return provider
    
    # Check for provider name in the model string (fallback)
    for provider in providers.keys():
        if provider.lower() in model.lower():
            return provider
    
    # List all supported models for the error message
    all_supported = []
    for provider, models in providers.items():
        all_supported.extend(models)
    
    write_log(f"WARNING: Unsupported model: {model}. Using provider name from model string.", 'warning')
    
    # Try to extract provider name from the model string
    for known_provider in ['openai', 'anthropic', 'claude', 'gpt', 'deepseek', 'gemini', 
                          'google', 'qwen', 'alibaba', 'step', 'glm', 'zhipu', 'minimax']:
        if known_provider in model.lower():
            if known_provider == 'gpt':
                return 'openai'
            elif known_provider == 'claude':
                return 'anthropic'
            elif known_provider == 'google':
                return 'gemini'
            elif known_provider == 'alibaba':
                return 'qwen'
            elif known_provider == 'glm':
                return 'zhipu'
            else:
                return known_provider
                
    # If we still can't determine the provider, raise an error
    raise ValueError(f"Unsupported model: {model}. Supported models are: {', '.join(all_supported)}")


def select_best_prediction(predictions: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Select the best prediction from multiple models.
    
    Args:
        predictions: List of dictionaries mapping cluster IDs to cell type annotations
        
    Returns:
        Dict[str, str]: Dictionary mapping cluster IDs to best predictions
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


def identify_controversial_clusters(annotations: Dict[str, Dict[str, str]], 
                                  threshold: float = 0.6) -> List[str]:
    """
    Identify clusters with inconsistent annotations across models.
    
    Args:
        annotations: Dictionary mapping model names to dictionaries of cluster annotations
        threshold: Agreement threshold below which a cluster is considered controversial
        
    Returns:
        List[str]: List of controversial cluster IDs
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
        for model, results in annotations.items():
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


def annotate_cell_types(
    input_data: Union[pd.DataFrame, List[str], Dict[str, List[str]]],
    tissue_name: str = None,
    model: str = 'gpt-4o',
    api_key: str = None,
    top_gene_number: int = 10,
    use_cache: bool = True,
    format_json: bool = False
) -> LLMResponse:
    """
    Annotate cell types using various Large Language Models (LLMs).

    Args:
        input_data: Either a pandas DataFrame (from scanpy/Seurat FindAllMarkers), 
                   a list of genes, or a dictionary of gene lists
        tissue_name: Optional tissue name for context
        model: The LLM model to use
        api_key: API key for the selected model. If None, returns the prompt only
        top_gene_number: Number of top differential genes to use
        use_cache: Whether to use cached results
        format_json: Whether to request JSON formatted response

    Returns:
        LLMResponse object containing cell type predictions and the prompt used
    """
    # Process input data
    if isinstance(input_data, dict):
        processed_input = {k: ','.join(v) for k, v in input_data.items()}
    elif isinstance(input_data, list):
        processed_input = {'group1': ','.join(input_data)}
    elif isinstance(input_data, pd.DataFrame):
        # Filter for positive fold changes if the column exists
        if 'avg_log2FC' in input_data.columns:
            input_data = input_data[input_data['avg_log2FC'] > 0]
        
        # Group by cluster and get top genes
        def get_top_genes(group):
            return ','.join(group['gene'].head(top_gene_number))
        
        processed_input = input_data.groupby('cluster').apply(get_top_genes).to_dict()
    else:
        raise ValueError("Input must be a DataFrame, list, or dictionary")

    # Create prompt
    prompt_lines = [
        f"Identify cell types of {tissue_name} cells using the following markers separately for each",
        "row. Only provide the cell type name. Do not show numbers before the name.",
        "Some can be a mixture of multiple cell types."
    ]
    
    # Add JSON format instruction if requested
    if format_json:
        prompt_lines.append("")
        prompt_lines.append("Format your response as a JSON object with the following structure:")
        prompt_lines.append("""{
  "annotations": [
    {
      "cluster": "1",
      "cell_type": "T cells",
      "confidence": "high",
      "key_markers": ["CD3D", "CD3G", "CD3E"]
    },
    ...
  ]
}""")
    
    # Add cluster marker genes
    for key, genes in processed_input.items():
        prompt_lines.append(f"{key}:{genes}")
    
    prompt = '\n'.join(prompt_lines)

    # If no API key, return prompt only
    if api_key is None:
        write_log("Note: No API key provided: returning the prompt only")
        return LLMResponse(cell_types=[], prompt=prompt)

    # Get provider from model name
    provider = get_provider(model)
    write_log(f"Using provider: {provider} with model: {model}")

    # Create a unique cache key for this request if using cache
    cache_key = None
    if use_cache:
        from .utils import create_cache_key, load_from_cache
        cache_key = create_cache_key(prompt, model, provider)
        cached_result = load_from_cache(cache_key)
        if cached_result:
            write_log("Using cached result")
            cluster_ids = list(processed_input.keys())
            
            # Format the cached result
            from .utils import format_results
            formatted_result = format_results(cached_result, cluster_ids)
            result_list = [formatted_result.get(str(cluster_id), "Unknown") for cluster_id in cluster_ids]
            
            return LLMResponse(
                cell_types=result_list, 
                prompt=prompt,
                raw_response='\n'.join(cached_result),
                metadata={"provider": provider, "model": model, "cached": True}
            )

    # Process based on provider
    provider_map = {
        'openai': process_openai_legacy,
        'anthropic': process_anthropic_legacy,
        'deepseek': process_deepseek_legacy,
        'gemini': process_gemini_legacy,
        'qwen': process_qwen_legacy,
        'stepfun': process_stepfun_legacy,
        'zhipu': process_zhipu_legacy,
        'minimax': process_minimax_legacy
    }
    
    # Check if provider is supported
    if provider not in provider_map:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Get provider function
    provider_func = provider_map[provider]
    
    try:
        # Call provider function
        result = provider_func(prompt, model, api_key)
        
        # Save to cache if using cache
        if use_cache and cache_key:
            from .utils import save_to_cache
            save_to_cache(cache_key, result)
        
        write_log('Note: It is always recommended to check the results returned by LLMs in case of '
                 'AI hallucination, before proceeding with downstream analysis.')
        
        return LLMResponse(
            cell_types=result, 
            prompt=prompt,
            raw_response='\n'.join(result) if isinstance(result, list) else str(result),
            metadata={"provider": provider, "model": model, "cached": False}
        )
    except Exception as e:
        write_log(f"Error during LLM annotation: {str(e)}", level='error')
        raise


def process_openai_legacy(prompt: str, model: str, api_key: str) -> List[str]:
    """Process request using OpenAI models (legacy function)"""
    write_log(f"Using OpenAI API with model: {model}")
    
    # Initialize OpenAI client
    openai.api_key = api_key
    
    # Split into chunks of 30 like the R version
    input_lines = prompt.split('\n')
    chunk_size = 30
    chunks = [input_lines[i:i + chunk_size] for i in range(0, len(input_lines), chunk_size)]
    
    all_results = []
    for i, chunk in enumerate(chunks):
        chunk_text = '\n'.join(chunk)
        write_log(f"Processing chunk {i+1} of {len(chunks)}")
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": chunk_text}]
                )
                result = response.choices[0].message.content.strip().split('\n')
                
                # Verify we got the expected number of responses
                expected_lines = len(chunk) - 3  # -3 for the header lines
                if len(result) >= expected_lines:
                    # If we got more lines than expected, take the first expected_lines
                    result = result[:expected_lines]
                    all_results.extend(result)
                    break
                else:
                    # If we didn't get enough lines, retry
                    write_log(f"WARNING: Expected {expected_lines} lines, got {len(result)}", level='warning')
                    if attempt < max_retries - 1:
                        write_log(f"Retrying (attempt {attempt+2}/{max_retries})...")
                        time.sleep(retry_delay)
                    else:
                        # If this is the last attempt, use what we got
                        all_results.extend(result)
                        # Pad with "Unknown" to match expected length
                        all_results.extend(["Unknown"] * (expected_lines - len(result)))
            except Exception as e:
                write_log(f"Error during OpenAI API call (attempt {attempt+1}/{max_retries}): {str(e)}", level='error')
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    write_log(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    raise
    
    return [r.rstrip(',') for r in all_results]


def process_anthropic_legacy(prompt: str, model: str, api_key: str) -> List[str]:
    """Process request using Anthropic models (legacy function)"""
    write_log(f"Using Anthropic API with model: {model}")
    
    # Import here to avoid dependency if not using Anthropic
    try:
        import anthropic
        
        # Create a client
        client = anthropic.Anthropic(
            api_key=api_key
        )
        
        # Get the model to use
        if not model or model == "default":
            model = "claude-3-opus-20240229"
        
        write_log(f"Sending request to Anthropic API with model: {model}")
        
        # Call the API
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the result
        result = response.content[0].text.strip().split('\n')
        write_log(f"Received response from Anthropic API with {len(result)} lines")
        
        # Count number of clusters in prompt
        cluster_count = len(prompt.split('\n')) - 3  # -3 for header lines
        
        # If we got fewer lines than clusters, pad with "Unknown"
        if len(result) < cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}", level='warning')
            result.extend(["Unknown"] * (cluster_count - len(result)))
        
        # If we got more lines than clusters, truncate
        if len(result) > cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}, truncating", level='warning')
            result = result[:cluster_count]
        
        return result
    except ImportError:
        raise ImportError("Anthropic package not installed. Please install with 'pip install anthropic'.")
    except Exception as e:
        write_log(f"Error during Anthropic API call: {str(e)}", level='error')
        raise


def process_deepseek_legacy(prompt: str, model: str, api_key: str) -> List[str]:
    """Process request using DeepSeek models (legacy function)"""
    write_log(f"Using DeepSeek API with model: {model}")
    
    try:
        # URL for DeepSeek API
        url = "https://api.deepseek.com/v1/chat/completions"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "model": model if model else "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000
        }
        
        # Make the API call
        write_log("Sending request to DeepSeek API")
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            write_log(f"DeepSeek API error: {response.status_code} - {response.text}", level='error')
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        # Parse response
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Split into lines
        result = content.strip().split('\n')
        write_log(f"Received response from DeepSeek API with {len(result)} lines")
        
        # Count number of clusters in prompt
        cluster_count = len(prompt.split('\n')) - 3  # -3 for header lines
        
        # If we got fewer lines than clusters, pad with "Unknown"
        if len(result) < cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}", level='warning')
            result.extend(["Unknown"] * (cluster_count - len(result)))
            
        # If we got more lines than clusters, truncate
        if len(result) > cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}, truncating", level='warning')
            result = result[:cluster_count]
        
        return result
    except Exception as e:
        write_log(f"Error during DeepSeek API call: {str(e)}", level='error')
        raise


def process_gemini_legacy(prompt: str, model: str, api_key: str) -> List[str]:
    """Process request using Gemini models (legacy function)"""
    write_log(f"Using Gemini API with model: {model}")
    
    try:
        # Try to import the Google Generative AI library
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Please install with 'pip install google-generativeai'.")
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Set the model
        if not model or model == "default":
            model = "gemini-2.0-pro"
        
        # Create the model
        model_obj = genai.GenerativeModel(model_name=model)
        
        # Generate content
        write_log("Sending request to Gemini API")
        response = model_obj.generate_content(prompt)
        
        # Extract the result
        content = response.text
        result = content.strip().split('\n')
        write_log(f"Received response from Gemini API with {len(result)} lines")
        
        # Count number of clusters in prompt
        cluster_count = len(prompt.split('\n')) - 3  # -3 for header lines
        
        # If we got fewer lines than clusters, pad with "Unknown"
        if len(result) < cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}", level='warning')
            result.extend(["Unknown"] * (cluster_count - len(result)))
            
        # If we got more lines than clusters, truncate
        if len(result) > cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}, truncating", level='warning')
            result = result[:cluster_count]
        
        return result
    except Exception as e:
        write_log(f"Error during Gemini API call: {str(e)}", level='error')
        raise


def process_qwen_legacy(prompt: str, model: str, api_key: str) -> List[str]:
    """Process request using Qwen models (legacy function)"""
    write_log(f"Using Qwen API with model: {model}")
    
    try:
        # URL for Qwen API (Alibaba Cloud DashScope)
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "model": model if model else "qwen-max-2025-01-25",
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "max_tokens": 4000
            }
        }
        
        # Make the API call
        write_log("Sending request to Qwen API")
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            write_log(f"Qwen API error: {response.status_code} - {response.text}", level='error')
            raise Exception(f"Qwen API error: {response.status_code} - {response.text}")
        
        # Parse response
        response_data = response.json()
        content = response_data.get("output", {}).get("text", "")
        
        # Split into lines
        result = content.strip().split('\n')
        write_log(f"Received response from Qwen API with {len(result)} lines")
        
        # Count number of clusters in prompt
        cluster_count = len(prompt.split('\n')) - 3  # -3 for header lines
        
        # If we got fewer lines than clusters, pad with "Unknown"
        if len(result) < cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}", level='warning')
            result.extend(["Unknown"] * (cluster_count - len(result)))
            
        # If we got more lines than clusters, truncate
        if len(result) > cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}, truncating", level='warning')
            result = result[:cluster_count]
        
        return result
    except Exception as e:
        write_log(f"Error during Qwen API call: {str(e)}", level='error')
        raise


def process_stepfun_legacy(prompt: str, model: str, api_key: str) -> List[str]:
    """Process request using Stepfun models (legacy function)"""
    write_log(f"Using Stepfun API with model: {model}")
    
    try:
        # URL for Stepfun API
        url = "https://api.stepfun.com/v1/chat/completions"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "model": model if model else "step-2-16k",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000
        }
        
        # Make the API call
        write_log("Sending request to Stepfun API")
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            write_log(f"Stepfun API error: {response.status_code} - {response.text}", level='error')
            raise Exception(f"Stepfun API error: {response.status_code} - {response.text}")
        
        # Parse response
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Split into lines
        result = content.strip().split('\n')
        write_log(f"Received response from Stepfun API with {len(result)} lines")
        
        # Count number of clusters in prompt
        cluster_count = len(prompt.split('\n')) - 3  # -3 for header lines
        
        # If we got fewer lines than clusters, pad with "Unknown"
        if len(result) < cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}", level='warning')
            result.extend(["Unknown"] * (cluster_count - len(result)))
            
        # If we got more lines than clusters, truncate
        if len(result) > cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}, truncating", level='warning')
            result = result[:cluster_count]
        
        return result
    except Exception as e:
        write_log(f"Error during Stepfun API call: {str(e)}", level='error')
        raise


def process_zhipu_legacy(prompt: str, model: str, api_key: str) -> List[str]:
    """Process request using Zhipu models (legacy function)"""
    write_log(f"Using Zhipu API with model: {model}")
    
    try:
        # URL for Zhipu API
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "model": model if model else "glm-4",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000
        }
        
        # Make the API call
        write_log("Sending request to Zhipu API")
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            write_log(f"Zhipu API error: {response.status_code} - {response.text}", level='error')
            raise Exception(f"Zhipu API error: {response.status_code} - {response.text}")
        
        # Parse response
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Split into lines
        result = content.strip().split('\n')
        write_log(f"Received response from Zhipu API with {len(result)} lines")
        
        # Count number of clusters in prompt
        cluster_count = len(prompt.split('\n')) - 3  # -3 for header lines
        
        # If we got fewer lines than clusters, pad with "Unknown"
        if len(result) < cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}", level='warning')
            result.extend(["Unknown"] * (cluster_count - len(result)))
            
        # If we got more lines than clusters, truncate
        if len(result) > cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}, truncating", level='warning')
            result = result[:cluster_count]
        
        return result
    except Exception as e:
        write_log(f"Error during Zhipu API call: {str(e)}", level='error')
        raise


def process_minimax_legacy(prompt: str, model: str, api_key: str) -> List[str]:
    """Process request using MiniMax models (legacy function)"""
    write_log(f"Using MiniMax API with model: {model}")
    
    try:
        # URL for MiniMax API
        url = "https://api.minimax.chat/v1/text/completion"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "model": model if model else "minimax-text-01",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000
        }
        
        # Make the API call
        write_log("Sending request to MiniMax API")
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            write_log(f"MiniMax API error: {response.status_code} - {response.text}", level='error')
            raise Exception(f"MiniMax API error: {response.status_code} - {response.text}")
        
        # Parse response
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Split into lines
        result = content.strip().split('\n')
        write_log(f"Received response from MiniMax API with {len(result)} lines")
        
        # Count number of clusters in prompt
        cluster_count = len(prompt.split('\n')) - 3  # -3 for header lines
        
        # If we got fewer lines than clusters, pad with "Unknown"
        if len(result) < cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}", level='warning')
            result.extend(["Unknown"] * (cluster_count - len(result)))
            
        # If we got more lines than clusters, truncate
        if len(result) > cluster_count:
            write_log(f"WARNING: Expected {cluster_count} lines, got {len(result)}, truncating", level='warning')
            result = result[:cluster_count]
        
        return result
    except Exception as e:
        write_log(f"Error during MiniMax API call: {str(e)}", level='error')
        raise
