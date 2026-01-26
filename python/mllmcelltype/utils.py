"""Utility functions for LLMCellType."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Optional, Union

import pandas as pd

from .logger import write_log


def _get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Get cache directory path with consistent handling.

    Args:
        cache_dir: Custom cache directory or None for default

    Returns:
        str: Cache directory path
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".mllmcelltype", "cache")
    return cache_dir


def load_api_key(provider: str) -> str:
    """Load API key for a specific provider from environment variables or .env file.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')

    Returns:
        str: The API key

    """
    # Map of provider names to environment variable names
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",  # Changed from GOOGLE_API_KEY to match .env file
        "qwen": "QWEN_API_KEY",  # Changed from DASHSCOPE_API_KEY to match .env file
        "stepfun": "STEPFUN_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "grok": "GROK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    # Get environment variable name for provider
    env_var = env_var_map.get(provider.lower())
    if not env_var:
        write_log(f"Unknown provider: {provider}", level="warning")
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
                potential_path = os.path.join(current_dir, ".env")
                if os.path.isfile(potential_path):
                    env_path = potential_path
                    break
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Reached root directory
                    break
                current_dir = parent_dir

            # Also check for a .env file in the package directory
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            package_env_path = os.path.join(package_dir, ".env")
            if os.path.isfile(package_env_path):
                env_path = package_env_path

            if env_path:
                write_log(f"Found .env file at {env_path}")
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            key, value = line.split("=", 1)
                            if key == env_var:
                                api_key = value
                                write_log(f"Loaded API key for {provider} from .env file")
                                break
        except (OSError, ValueError) as e:
            write_log(f"Error loading .env file: {str(e)}", level="warning")

    if not api_key:
        write_log(f"API key not found for provider: {env_var}", level="warning")

    return api_key


def create_cache_key(prompt: str, model: str, provider: str) -> str:
    """Create a cache key for a specific request.

    Args:
        prompt: The prompt text
        model: The model name
        provider: The provider name

    Returns:
        str: The cache key

    """
    # Normalize inputs to ensure consistent keys
    normalized_provider = str(provider).lower().strip()
    normalized_model = str(model).lower().strip()
    normalized_prompt = str(prompt).strip()

    # For OpenRouter models (containing '/'), ensure provider is 'openrouter'
    # This prevents cache key collisions between different providers
    if "/" in normalized_model:
        normalized_provider = "openrouter"

    # Create a string to hash with clear separators to avoid collisions
    hash_string = (
        f"provider:{normalized_provider}||model:{normalized_model}||prompt:{normalized_prompt}"
    )

    # Create hash
    hash_object = hashlib.sha256(hash_string.encode("utf-8"))
    return hash_object.hexdigest()


def save_to_cache(
    cache_key: str,
    results: Union[list[str], dict[str, Any]],
    cache_dir: Optional[str] = None,
) -> None:
    """Save results to cache.

    Args:
        cache_key: The cache key
        results: The results to cache (list of strings or dictionary)
        cache_dir: The cache directory. If None, uses default directory.

    """
    cache_dir = _get_cache_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Ensure results are in a consistent format
    cache_data = {"version": "1.0", "timestamp": time.time(), "data": results}

    # Save results to cache
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        write_log(f"Saved results to cache: {cache_file}")
    except (OSError, TypeError, ValueError) as e:
        write_log(f"Error saving cache for {cache_file}: {e}", level="error")


def load_from_cache(
    cache_key: str, cache_dir: Optional[str] = None
) -> Optional[Union[list[str], dict[str, Any]]]:
    """Load results from cache.

    Args:
        cache_key: The cache key
        cache_dir: The cache directory. If None, uses default directory.

    Returns:
        Optional[Union[list[str], dict[str, Any]]]: The cached results, or None if not found

    """
    cache_dir = _get_cache_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    # Check if cache file exists
    if not os.path.exists(cache_file):
        return None

    # Load results from cache
    try:
        with open(cache_file) as f:
            cache_content = json.load(f)

        # Handle different cache formats
        if isinstance(cache_content, dict) and "data" in cache_content:
            # New format with metadata
            results = cache_content["data"]
            write_log(
                f"Loaded results from cache (version {cache_content.get('version', 'unknown')}): {cache_file}"
            )
        else:
            # Old format (direct data)
            results = cache_content
            write_log(f"Loaded results from cache (legacy format): {cache_file}")

        return results
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        write_log(f"Error loading cache for {cache_file}: {e}", level="warning")
        return None


def parse_marker_genes(marker_genes_df: pd.DataFrame) -> dict[str, list[str]]:
    """Parse marker genes dataframe into a dictionary.

    Args:
        marker_genes_df: DataFrame containing marker genes

    Returns:
        dict[str, list[str]]: Dictionary mapping cluster names to lists of marker genes

    """
    result = {}

    # Check if dataframe is empty
    if marker_genes_df.empty:
        write_log("Empty marker genes dataframe", level="warning")
        return result

    # Get column names
    columns = marker_genes_df.columns.tolist()

    # Check if 'cluster' column exists
    if "cluster" not in columns:
        write_log("'cluster' column not found in marker genes dataframe", level="error")
        raise ValueError("'cluster' column not found in marker genes dataframe")

    # Check if 'gene' column exists
    if "gene" not in columns:
        write_log("'gene' column not found in marker genes dataframe", level="error")
        raise ValueError("'gene' column not found in marker genes dataframe")

    # Group by cluster and get list of genes
    for cluster, group in marker_genes_df.groupby("cluster"):
        result[str(cluster)] = group["gene"].tolist()

    return result


def format_results(results: list[str], clusters: list[str]) -> dict[str, str]:
    """Format results into a dictionary mapping cluster names to annotations.

    Args:
        results: List of annotation results (one line per cluster)
        clusters: List of cluster names

    Returns:
        dict[str, str]: Dictionary mapping cluster names to annotations

    """
    # Clean up results (remove empty lines and whitespace)
    clean_results = [line.strip() for line in results if line.strip()]

    # Case 1: Try to parse the format "Cluster X: Annotation" (most common format from our prompts)
    result = {}
    cluster_pattern = r"Cluster\s+(\d+):\s*(.*)"

    # First pass: try to find annotations for each cluster by ID
    for cluster in clusters:
        cluster_str = str(cluster)

        # Extract numeric ID from cluster name (e.g., "Cluster_0" -> "0", "0" -> "0")
        cluster_id_match = re.search(r"(\d+)", cluster_str)
        cluster_id = cluster_id_match.group(1) if cluster_id_match else cluster_str

        # Look for exact matches (e.g., "Cluster 0: T cells")
        for line in clean_results:
            match = re.match(cluster_pattern, line)
            if match and match.group(1) == cluster_id:
                result[cluster_str] = match.group(2).strip()
                break

    # If we found annotations for all clusters, return the result
    if len(result) == len(clusters):
        write_log(
            "Successfully parsed response in 'Cluster X: Annotation' format",
            level="info",
        )
        return result

    # Case 2: Try to parse JSON response
    try:
        # Join all lines and try to find JSON content
        full_text = "\n".join(clean_results)

        # Extract JSON content if it's wrapped in ```json and ``` markers
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", full_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code blocks, try to find JSON object directly
            json_match = re.search(r"(\{[\s\S]*\})", full_text)
            # Extract JSON content or use full text
            json_str = json_match.group(1) if json_match else full_text

        # Fix common JSON formatting issues
        # Add missing commas between JSON objects
        json_str = re.sub(r'("[^"]+")\s*\n\s*("[^"]+")', r"\1,\n\2", json_str)
        # Add missing commas after closing brackets
        json_str = re.sub(r'(\])\s*\n\s*("[^"]+")', r"\1,\n\2", json_str)
        # Add missing commas after closing braces
        json_str = re.sub(r'(\})\s*\n\s*("[^"]+")', r"\1,\n\2", json_str)
        # Add missing commas after closing braces before opening braces
        json_str = re.sub(r"(\})\s*\n\s*(\{)", r"\1,\n\2", json_str)

        # Parse JSON
        data = json.loads(json_str)

        # Extract annotations from JSON structure
        if "annotations" in data and isinstance(data["annotations"], list):
            json_result = {}

            for annotation in data["annotations"]:
                if "cluster" in annotation and "cell_type" in annotation:
                    cluster_id = annotation["cluster"]
                    json_result[cluster_id] = annotation["cell_type"]

            # If we found annotations for all clusters, return the result
            if len(json_result) == len(clusters):
                write_log("Successfully parsed JSON response", level="info")
                return json_result
    except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:
        write_log(f"Failed to parse JSON response: {str(e)}", level="debug")

    # Case 3: Check if this is a simple response where each line corresponds to a cluster
    # This is the expected format from the R version
    if len(clean_results) >= len(clusters):
        # Simple case: one result per cluster
        simple_result = {}
        for i, cluster in enumerate(clusters):
            if i < len(clean_results):
                # Check if this line contains a cluster prefix and remove it
                line = clean_results[i]
                match = re.match(cluster_pattern, line)
                if match:
                    simple_result[str(cluster)] = match.group(2).strip()
                else:
                    simple_result[str(cluster)] = line.strip()
            else:
                simple_result[str(cluster)] = "Unknown"

        write_log("Successfully parsed response as simple line-by-line format", level="info")
        return simple_result

    # Case 4: Fall back to the original method
    write_log(
        "Could not parse complex LLM response, falling back to simple mapping",
        level="warning",
    )
    result = {}
    for i, cluster in enumerate(clusters):
        if i < len(clean_results):
            result[str(cluster)] = clean_results[i]
        else:
            result[str(cluster)] = "Unknown"

    # Check if number of results matches number of clusters
    if len(result) != len(clusters):
        write_log(
            f"Number of results ({len(result)}) does not match number of clusters ({len(clusters)})",
            level="warning",
        )

    return result


def clean_annotation(annotation: str) -> str:
    """Clean up cell type annotation from LLM response.

    Args:
        annotation: Raw annotation string

    Returns:
        str: Cleaned annotation

    """
    # If input is empty or None, return an empty string
    if not annotation:
        return ""

    # Basic cleanup
    annotation = annotation.strip()

    # Step 1: Remove common prefixes
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
            annotation = annotation[len(prefix) :].strip()

    # Step 2: Extract cell type from descriptive text
    patterns = [
        r"([\w\s-]+)\s+(?:is|are)\s+the\s+most\s+accurate\s+cell\s+type",
        r"([\w\s-]+)\s+(?:is|are)\s+the\s+best\s+annotation",
        r"final\s+cell\s+type\s*:?\s*([\w\s-]+)",
        r"final\s+decision\s*:?\s*([\w\s-]+)",
        r"majority\s+prediction\s*:?\s*([\w\s-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, annotation.lower())
        if match:
            annotation = match.group(1).strip()
            break

    # Step 3: Clean formatting marks and symbols
    # Remove quotes
    if annotation.startswith('"') and annotation.endswith('"'):
        annotation = annotation[1:-1]

    # Remove markdown emphasis marks
    annotation = annotation.replace("**:", "").replace("**", "").replace("*", "")

    # Remove common prefix markers
    if annotation.startswith("-"):
        annotation = annotation[1:].strip()

    # Remove prefixes like "Final Cell Type:"
    if ":" in annotation and any(
        x in annotation.lower() for x in ["final", "type", "determination", "conclusion"]
    ):
        parts = annotation.split(":", 1)
        annotation = parts[1].strip()

    # Remove trailing punctuation
    if annotation and annotation[-1] in [".", ",", ";"]:
        annotation = annotation[:-1]

    # Remove LaTeX formatting
    annotation = re.sub(r"\$\\boxed\{(.+?)\}\$", r"\1", annotation)
    annotation = re.sub(r"\$.+?\$", "", annotation)

    # Truncate long descriptions
    if len(annotation) > 50:
        short_version = annotation.split(",")[0].split("(")[0].strip()
        if len(short_version) >= 10:
            annotation = short_version

    return annotation


def normalize_annotation_for_comparison(annotation: str) -> str:
    """Normalize cell type annotation for comparison purposes.

    This function mirrors the R implementation's normalize_annotation function:
    1. Converts to lowercase
    2. Trims whitespace
    3. Converts multiple spaces to single space
    4. Removes punctuation
    5. Handles plurals (cells→cell, cytes→cyte) - using singular form for consistency
    6. Applies specific cell type mappings

    Args:
        annotation: Cell type annotation string

    Returns:
        str: Normalized annotation for comparison
    """
    if not annotation:
        return ""

    # Start with basic cleaning
    normalized = clean_annotation(annotation)

    # Convert to lowercase for comparison (mirrors R's tolower)
    normalized = normalized.lower()

    # Trim whitespace (mirrors R's trimws)
    normalized = normalized.strip()

    # Convert multiple spaces to single (mirrors R's gsub("\\s+", " ", ...))
    normalized = " ".join(normalized.split())

    # Remove punctuation (mirrors R's gsub("[[:punct:]]", "", ...))
    # But preserve '+' as it's meaningful for cell markers
    # Replace with space first to avoid joining words
    normalized = re.sub(r"[^\w\s+]", " ", normalized)
    normalized = " ".join(normalized.split())

    # Handle plurals - convert to singular for consistency (mirrors R's logic)
    # cells → cell, cytes → cyte
    normalized = re.sub(r"cells\b", "cell", normalized)
    normalized = re.sub(r"cytes\b", "cyte", normalized)

    # Common cell type variations (mirrors R's specific mappings)
    # t lymphocyte → t cell
    normalized = re.sub(r"\bt lymphocyte\b", "t cell", normalized)
    # b lymphocyte → b cell
    normalized = re.sub(r"\bb lymphocyte\b", "b cell", normalized)
    # natural killer → nk
    normalized = re.sub(r"\bnatural killer\b", "nk", normalized)

    # Additional normalizations for common variations
    # naive/naïve normalization
    normalized = normalized.replace("naïve", "naive")
    # tumour/tumor normalization
    normalized = normalized.replace("tumour", "tumor")

    # Handle CD markers - normalize spacing
    # cd4+ t cell, cd4 positive t cell → cd4+ t cell
    normalized = re.sub(r"\bcd(\d+)\s+positive\b", r"cd\1+", normalized)
    # cd 4 → cd4
    normalized = re.sub(r"\bcd\s+(\d+)", r"cd\1", normalized)

    # Remove extra whitespace again after all transformations
    normalized = " ".join(normalized.split())

    return normalized


def clear_cache(cache_dir: Optional[str] = None, older_than: Optional[int] = None) -> int:
    """Clear cache.

    Args:
        cache_dir: Cache directory
        older_than: Only clear items older than this many seconds.
                   If None, clear all cache.

    Returns:
        int: Number of cache files removed

    """
    cache_dir = _get_cache_dir(cache_dir)

    if not os.path.exists(cache_dir):
        return 0

    # Get all cache files
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]

    if not older_than:
        # Remove all cache files
        count = 0
        for f in cache_files:
            try:
                os.remove(os.path.join(cache_dir, f))
                count += 1
            except OSError as e:
                write_log(f"Error removing cache file {f}: {e}", level="warning")
        return count
    # Remove only older files
    now = time.time()
    count = 0
    for f in cache_files:
        file_path = os.path.join(cache_dir, f)
        try:
            # Check file age using metadata
            with open(file_path) as file:
                cache_data = json.load(file)

            # Handle different cache formats
            if isinstance(cache_data, dict) and "timestamp" in cache_data:
                # New format with metadata
                file_age = now - cache_data.get("timestamp", 0)
            else:
                # Legacy format - use file modification time
                file_age = now - os.path.getmtime(file_path)

            if file_age > older_than:
                os.remove(file_path)
                count += 1
        except OSError as e:
            write_log(f"Error processing cache file {f}: {e}", level="warning")

    return count


def get_cache_stats(cache_dir: Optional[str] = None, detailed: bool = True) -> dict[str, Any]:
    """Get cache statistics.

    Args:
        cache_dir: The cache directory (None for default)
        detailed: If True, read file contents for detailed stats (slower).
                  If False, only return basic file counts and sizes (faster).

    Returns:
        dict[str, Any]: Cache statistics

    """
    cache_dir = _get_cache_dir(cache_dir)

    if not os.path.exists(cache_dir):
        result = {
            "exists": False,
            "path": cache_dir,
            "count": 0,
            "size": 0,
            "size_mb": 0.0,
        }
        if detailed:
            result.update({
                "status": "No cache directory",
                "oldest": None,
                "newest": None,
            })
        return result

    # Get all cache files
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)

    # Basic stats (fast path)
    result = {
        "exists": True,
        "path": cache_dir,
        "count": len(cache_files),
        "size": total_size,
        "size_mb": total_size / (1024 * 1024),
    }

    if not detailed:
        return result

    # Detailed stats (slow path - reads file contents)
    if not cache_files:
        result.update({
            "status": "Empty cache",
            "oldest": None,
            "newest": None,
            "format_counts": {"legacy": 0, "1.0": 0, "unknown": 0},
            "valid_files": 0,
            "invalid_files": 0,
        })
        return result

    oldest = float("inf")
    newest = 0
    format_counts = {"legacy": 0, "1.0": 0, "unknown": 0}
    valid_files = 0
    invalid_files = 0

    for f in cache_files:
        file_path = os.path.join(cache_dir, f)
        try:
            # Load cache data
            with open(file_path) as file:
                cache_data = json.load(file)

            valid_files += 1

            # Check cache format
            if isinstance(cache_data, dict):
                if "version" in cache_data and "data" in cache_data:
                    # New format with metadata
                    version = cache_data.get("version", "unknown")
                    format_counts[version if version in format_counts else "unknown"] += 1

                    # Get timestamp
                    timestamp = cache_data.get("timestamp", 0)
                    oldest = min(oldest, timestamp)
                    newest = max(newest, timestamp)
                else:
                    # Legacy format or other dict format
                    format_counts["legacy"] += 1
            else:
                # Unknown format
                format_counts["unknown"] += 1

        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
        ) as e:
            invalid_files += 1
            write_log(f"Error processing cache file {f}: {e}", level="warning")

    # Convert timestamps to readable format
    oldest_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(oldest)) if oldest != float("inf") else None
    newest_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest)) if newest != 0 else None

    result.update({
        "status": "Cache available",
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "oldest": oldest_str,
        "newest": newest_str,
        "format_counts": format_counts,
    })

    return result
