"""
Utility functions for LLMCellType.
"""

import os
import re
import json
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging
import time
import jsonschema
from jsonschema.exceptions import ValidationError

from .logger import write_log

def load_api_key(provider: str) -> str:
    """
    Load API key for a specific provider from environment variables or .env file.
    
    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')
        
    Returns:
        str: The API key
    """
    # Map of provider names to environment variable names
    env_var_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'gemini': 'GEMINI_API_KEY',  # Changed from GOOGLE_API_KEY to match .env file
        'qwen': 'QWEN_API_KEY',      # Changed from DASHSCOPE_API_KEY to match .env file
        'stepfun': 'STEPFUN_API_KEY',
        'zhipu': 'ZHIPU_API_KEY',
        'minimax': 'MINIMAX_API_KEY'
    }
    
    # Get environment variable name for provider
    env_var = env_var_map.get(provider.lower())
    if not env_var:
        write_log(f"WARNING: Unknown provider: {provider}", level='warning')
        env_var = f"{provider.upper()}_API_KEY"
    
    # Get API key from environment variable
    api_key = os.getenv(env_var)
    
    # If not found in environment, try to load from .env file
    if not api_key:
        try:
            # Try to find .env file in project root (current directory or parent directories)
            env_path = None
            current_dir = os.path.abspath(os.getcwd())
            
            # Check current directory and up to 3 parent directories
            for _ in range(4):
                potential_path = os.path.join(current_dir, '.env')
                if os.path.isfile(potential_path):
                    env_path = potential_path
                    break
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Reached root directory
                    break
                current_dir = parent_dir
            
            # Also check user's home directory for LLMCelltype directory
            home_dir = os.path.expanduser('~')
            research_path = os.path.join(home_dir, 'Research', 'LLMCelltype', '.env')
            if os.path.isfile(research_path):
                env_path = research_path
            
            if env_path:
                write_log(f"Found .env file at {env_path}")
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            if key == env_var:
                                api_key = value
                                write_log(f"Loaded API key for {provider} from .env file")
                                break
        except Exception as e:
            write_log(f"Error loading .env file: {str(e)}", level='warning')
    
    if not api_key:
        write_log(f"WARNING: API key not found for provider: {env_var}", level='warning')
    
    return api_key

def create_cache_key(prompt: str, model: str, provider: str) -> str:
    """
    Create a cache key for a specific request.
    
    Args:
        prompt: The prompt text
        model: The model name
        provider: The provider name
        
    Returns:
        str: The cache key
    """
    # Create a string to hash
    hash_string = f"{provider}_{model}_{prompt}"
    
    # Create hash
    hash_object = hashlib.md5(hash_string.encode())
    return hash_object.hexdigest()

def save_to_cache(cache_key: str, results: List[str], cache_dir: Optional[str] = None) -> None:
    """
    Save results to cache.
    
    Args:
        cache_key: The cache key
        results: The results to cache
        cache_dir: The cache directory. If None, uses default directory.
    """
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.llmcelltype', 'cache')
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache file path
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # Save results to cache
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    
    write_log(f"Saved results to cache: {cache_file}")

def load_from_cache(cache_key: str, cache_dir: Optional[str] = None) -> Optional[List[str]]:
    """
    Load results from cache.
    
    Args:
        cache_key: The cache key
        cache_dir: The cache directory. If None, uses default directory.
        
    Returns:
        Optional[List[str]]: The cached results, or None if not found
    """
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.llmcelltype', 'cache')
    
    # Create cache file path
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # Check if cache file exists
    if not os.path.exists(cache_file):
        return None
    
    # Load results from cache
    try:
        with open(cache_file, 'r') as f:
            results = json.load(f)
        
        write_log(f"Loaded results from cache: {cache_file}")
        return results
    except Exception as e:
        write_log(f"Error loading from cache: {str(e)}", level='error')
        return None

def parse_marker_genes(marker_genes_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Parse marker genes dataframe into a dictionary.
    
    Args:
        marker_genes_df: DataFrame containing marker genes
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping cluster names to lists of marker genes
    """
    result = {}
    
    # Check if dataframe is empty
    if marker_genes_df.empty:
        write_log("WARNING: Empty marker genes dataframe", level='warning')
        return result
    
    # Get column names
    columns = marker_genes_df.columns.tolist()
    
    # Check if 'cluster' column exists
    if 'cluster' not in columns:
        write_log("ERROR: 'cluster' column not found in marker genes dataframe", level='error')
        raise ValueError("'cluster' column not found in marker genes dataframe")
    
    # Check if 'gene' column exists
    if 'gene' not in columns:
        write_log("ERROR: 'gene' column not found in marker genes dataframe", level='error')
        raise ValueError("'gene' column not found in marker genes dataframe")
    
    # Group by cluster and get list of genes
    for cluster, group in marker_genes_df.groupby('cluster'):
        result[str(cluster)] = group['gene'].tolist()
    
    return result

def get_annotation_metadata(annotation_result: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve metadata for a specific annotation result.
    
    Args:
        annotation_result: Dictionary mapping cluster IDs to cell type annotations
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping cluster IDs to metadata
    """
    try:
        # Create a unique key for this annotation result
        key = hashlib.md5(str(annotation_result).encode()).hexdigest()
        
        # Check if metadata exists in cache
        cache_dir = os.path.expanduser("~/.llmcelltype/metadata")
        cache_file = os.path.join(cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            write_log(f"No metadata found for the given annotation result", level='debug')
            return {}
    except Exception as e:
        write_log(f"Failed to retrieve metadata: {str(e)}", level='debug')
        return {}

def format_results(results: List[str], clusters: List[str]) -> Dict[str, str]:
    """
    Format results into a dictionary mapping cluster names to annotations.
    
    Args:
        results: List of annotation results
        clusters: List of cluster names
        
    Returns:
        Dict[str, str]: Dictionary mapping cluster names to annotations
    """
    import re
    import json
    
    # Check if number of results matches number of clusters
    if len(results) != len(clusters):
        write_log(f"WARNING: Number of results ({len(results)}) does not match number of clusters ({len(clusters)})", level='warning')
    
    # Case 1: Try to parse JSON response
    try:
        # Join all lines and try to find JSON content
        full_text = "\n".join(results)
        
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
                json_str = full_text
        
        # Fix common JSON formatting issues
        # Add missing commas between JSON objects
        json_str = re.sub(r'("[^"]+")\s*\n\s*("[^"]+")', r'\1,\n\2', json_str)
        # Add missing commas after closing brackets
        json_str = re.sub(r'(\])\s*\n\s*("[^"]+")', r'\1,\n\2', json_str)
        # Add missing commas after closing braces
        json_str = re.sub(r'(\})\s*\n\s*("[^"]+")', r'\1,\n\2', json_str)
        # Add missing commas after closing braces before opening braces
        json_str = re.sub(r'(\})\s*\n\s*(\{)', r'\1,\n\2', json_str)
        
        write_log(f"Processed JSON string: {json_str}", level='debug')
        
        # Parse JSON
        data = json.loads(json_str)
        
        # Define the expected JSON schema
        json_schema = {
            "type": "object",
            "required": ["annotations"],
            "properties": {
                "annotations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["cluster", "cell_type"],
                        "properties": {
                            "cluster": {"type": "string"},
                            "cell_type": {"type": "string"},
                            "confidence": {"type": "string"},
                            "key_markers": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        
        # Validate JSON against schema
        try:
            jsonschema.validate(instance=data, schema=json_schema)
            write_log("JSON validation successful", level='debug')
        except ValidationError as e:
            write_log(f"JSON validation failed: {str(e)}", level='debug')
            # Continue anyway, we'll try to extract what we can
        
        # Extract annotations from JSON structure
        if "annotations" in data and isinstance(data["annotations"], list):
            result = {}
            metadata = {}
            
            for annotation in data["annotations"]:
                if "cluster" in annotation and "cell_type" in annotation:
                    cluster_id = annotation["cluster"]
                    result[cluster_id] = annotation["cell_type"]
                    
                    # Store additional metadata if available
                    cluster_metadata = {}
                    if "confidence" in annotation:
                        cluster_metadata["confidence"] = annotation["confidence"]
                    if "key_markers" in annotation:
                        cluster_metadata["key_markers"] = annotation["key_markers"]
                    
                    if cluster_metadata:
                        metadata[cluster_id] = cluster_metadata
            
            # If we found annotations for all clusters, return the result
            if len(result) == len(clusters):
                write_log("Successfully parsed JSON response", level='info')
                
                # Store metadata in cache for later retrieval if needed
                if metadata:
                    try:
                        cache_dir = os.path.expanduser("~/.llmcelltype/metadata")
                        os.makedirs(cache_dir, exist_ok=True)
                        
                        # Create a unique key for this annotation result
                        key = hashlib.md5(str(result).encode()).hexdigest()
                        cache_file = os.path.join(cache_dir, f"{key}.json")
                        
                        with open(cache_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                            
                        write_log(f"Stored annotation metadata to {cache_file}", level='debug')
                    except Exception as e:
                        write_log(f"Failed to store metadata: {str(e)}", level='debug')
                
                return result
    except Exception as e:
        write_log(f"Failed to parse JSON response: {str(e)}", level='debug')
        
        # Fallback: Try to extract annotations directly using regex
        try:
            result = {}
            # Look for patterns like "cluster": "1", "cell_type": "T cells"
            cluster_pattern = r'"cluster"\s*:\s*"(\d+)"[^}]*"cell_type"\s*:\s*"([^"]+)"'
            matches = re.findall(cluster_pattern, full_text)
            
            for cluster, cell_type in matches:
                result[cluster] = cell_type
                
            # If we found annotations for all clusters, return the result
            if len(result) == len(clusters):
                write_log("Successfully parsed JSON response using regex fallback", level='info')
                return result
        except Exception as e2:
            write_log(f"Failed to parse JSON response using regex fallback: {str(e2)}", level='debug')
    
    # Case 2: Check if this is a simple response where each line corresponds to a cluster
    if len(results) == len(clusters):
        # Simple case: one result per cluster
        result = {}
        for i, cluster in enumerate(clusters):
            result[str(cluster)] = results[i]
        return result
    
    # Case 3: Try to parse the format "Cluster X: Annotation"
    result = {}
    cluster_pattern = r"Cluster\s+(\d+):\s*(.*)"
    
    # Try to find annotations for each cluster
    for cluster in clusters:
        cluster_str = str(cluster)
        
        # Look for exact matches first (e.g., "Cluster 1: T cells")
        for line in results:
            match = re.match(cluster_pattern, line.strip())
            if match and match.group(1) == cluster_str:
                result[cluster_str] = match.group(2).strip()
                break
    
    # If we found annotations for all clusters, return the result
    if len(result) == len(clusters):
        return result
    
    # Case 4: Fall back to the original method
    write_log("WARNING: Could not parse complex LLM response, falling back to simple mapping", level='warning')
    result = {}
    for i, cluster in enumerate(clusters):
        if i < len(results):
            result[str(cluster)] = results[i]
        else:
            result[str(cluster)] = "No annotation available"
    
    return result

def clean_annotation(annotation: str) -> str:
    """
    Clean up cell type annotation from LLM response.
    
    Args:
        annotation: Raw annotation string
        
    Returns:
        str: Cleaned annotation
    """
    # Remove common prefixes and formatting
    annotation = annotation.strip()
    
    # Remove "Cluster X:" prefix if present
    if annotation.lower().startswith("cluster ") and ":" in annotation:
        annotation = annotation.split(":", 1)[1].strip()
        
    # Remove number prefix if present (e.g. "1. T cells" -> "T cells")
    if ". " in annotation and annotation[0].isdigit():
        parts = annotation.split(". ", 1)
        if parts[0].isdigit():
            annotation = parts[1]
    
    # Remove common prefixes
    prefixes = ["cell type:", "cell type", "annotation:", "annotation"]
    for prefix in prefixes:
        if annotation.lower().startswith(prefix):
            annotation = annotation[len(prefix):].strip()
    
    # Remove quotes
    if annotation.startswith('"') and annotation.endswith('"'):
        annotation = annotation[1:-1]
    
    # Remove trailing punctuation
    if annotation and annotation[-1] in ['.', ',', ';']:
        annotation = annotation[:-1]
        
    return annotation

def find_agreement(annotations: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Find the level of agreement between different model annotations.
    
    Args:
        annotations: Dictionary mapping model names to dictionaries of cluster annotations
        
    Returns:
        Tuple[Dict[str, str], Dict[str, float]]: Consensus annotations and confidence scores
    """
    consensus = {}
    confidence = {}
    
    # Ensure we have annotations
    if not annotations or not all(annotations.values()):
        return {}, {}
    
    # Get all clusters
    all_clusters = set()
    for model_results in annotations.values():
        all_clusters.update(model_results.keys())
    
    # Process each cluster
    for cluster in all_clusters:
        # Collect all annotations for this cluster
        cluster_annotations = []
        
        for model, results in annotations.items():
            if cluster in results:
                annotation = clean_annotation(results[cluster])
                if annotation:
                    cluster_annotations.append(annotation.lower())  # Convert to lowercase for case-insensitive comparison
        
        # Count occurrences of each annotation
        annotation_counts = {}
        for annotation in cluster_annotations:
            annotation_counts[annotation] = annotation_counts.get(annotation, 0) + 1
        
        # Find most common annotation
        if annotation_counts:
            most_common = max(annotation_counts.items(), key=lambda x: x[1])
            most_common_annotation = most_common[0]
            most_common_count = most_common[1]
            
            # Calculate confidence
            confidence_score = most_common_count / len(cluster_annotations) if cluster_annotations else 0
            
            consensus[cluster] = most_common_annotation
            confidence[cluster] = confidence_score
        else:
            consensus[cluster] = "Unknown"
            confidence[cluster] = 0.0
    
    return consensus, confidence

def clear_cache(cache_dir: Optional[str] = None, older_than: Optional[int] = None) -> int:
    """
    Clear cache.
    
    Args:
        cache_dir: Cache directory
        older_than: Only clear items older than this many seconds. 
                   If None, clear all cache.
                   
    Returns:
        int: Number of cache files removed
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.llmcelltype', 'cache')
    
    if not os.path.exists(cache_dir):
        return 0
    
    # Get all cache files
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    
    if not older_than:
        # Remove all cache files
        for f in cache_files:
            os.remove(os.path.join(cache_dir, f))
        return len(cache_files)
    else:
        # Remove only older files
        now = time.time()
        count = 0
        for f in cache_files:
            file_path = os.path.join(cache_dir, f)
            try:
                # Check file age using metadata
                with open(file_path, 'r') as file:
                    cache_data = json.load(file)
                file_age = now - cache_data.get('timestamp', 0)
                
                if file_age > older_than:
                    os.remove(file_path)
                    count += 1
            except Exception as e:
                write_log(f"Error processing cache file {f}: {e}", level='warning')
        
        return count

def get_cache_stats(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Args:
        cache_dir: The cache directory
        
    Returns:
        Dict[str, Any]: Cache statistics
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.llmcelltype', 'cache')
    
    if not os.path.exists(cache_dir):
        return {
            'status': 'No cache directory',
            'count': 0,
            'size': 0,
            'oldest': None,
            'newest': None
        }
    
    # Get all cache files
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    
    if not cache_files:
        return {
            'status': 'Empty cache',
            'count': 0,
            'size': 0,
            'oldest': None,
            'newest': None
        }
    
    # Calculate statistics
    total_size = 0
    oldest = float('inf')
    newest = 0
    provider_counts = {}
    
    for f in cache_files:
        file_path = os.path.join(cache_dir, f)
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            # Get timestamp from cache data
            with open(file_path, 'r') as file:
                cache_data = json.load(file)
            timestamp = cache_data.get('timestamp', 0)
            
            oldest = min(oldest, timestamp)
            newest = max(newest, timestamp)
            
            # Count by provider if available
            try:
                with open(file_path, 'r') as file:
                    cache_data = json.load(file)
                    if isinstance(cache_data, dict) and 'provider' in cache_data:
                        provider = cache_data['provider']
                        provider_counts[provider] = provider_counts.get(provider, 0) + 1
            except:
                pass
                
        except Exception as e:
            write_log(f"Error processing cache file {f}: {e}", level='warning')
    
    # Convert timestamps to readable format
    if oldest != float('inf'):
        oldest_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(oldest))
    else:
        oldest_str = None
        
    if newest != 0:
        newest_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest))
    else:
        newest_str = None
    
    return {
        'status': 'Cache available',
        'count': len(cache_files),
        'size': total_size,
        'size_readable': f"{total_size / (1024*1024):.2f} MB",
        'oldest': oldest_str,
        'newest': newest_str,
        'provider_counts': provider_counts
    }


def combine_results(model_results: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Combine results from multiple models into a single dictionary.
    
    Args:
        model_results: Dictionary mapping model names to dictionaries of cluster annotations
        
    Returns:
        Dict[str, Dict[str, str]]: Combined results
    """    
    # Simply return the model results as they are already in the correct format
    return model_results
