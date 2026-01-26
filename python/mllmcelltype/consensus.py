"""Module for consensus annotation of cell types from multiple LLM predictions."""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from collections import Counter
from typing import Any, Optional, Union

import requests

from .annotate import annotate_clusters, get_model_response
from .functions import get_provider
from .logger import write_log
from .prompts import (
    create_consensus_check_prompt,
    create_discussion_prompt,
    create_initial_discussion_prompt,
)
from .url_utils import resolve_provider_base_url
from .utils import clean_annotation, load_api_key, normalize_annotation_for_comparison

# Default fallback values when parsing fails or result is inconclusive
# These conservative values ensure discussion will be triggered
DEFAULT_FALLBACK_CONSENSUS_PROPORTION = 0.25
DEFAULT_FALLBACK_ENTROPY = 2.0

# Default result structure for discussion round consensus check
# Used when consensus check fails or has insufficient data
DEFAULT_CONSENSUS_RESULT = {
    "reached": False,
    "consensus_proportion": DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
    "entropy": DEFAULT_FALLBACK_ENTROPY,
    "majority_prediction": "Unknown",
}

# Default fallback LLM for consensus checking
DEFAULT_FALLBACK_PROVIDER = "anthropic"
DEFAULT_FALLBACK_MODEL = "claude-sonnet-4-5-20250929"


def _call_llm_with_retry(
    prompt: str,
    provider: str,
    model: str,
    api_key: Optional[str],
    max_retries: int = 3,
    fallback_provider: str = DEFAULT_FALLBACK_PROVIDER,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    api_keys: Optional[dict[str, str]] = None,
    base_urls: Optional[Union[str, dict[str, str]]] = None,
) -> Optional[str]:
    """Call LLM with retry logic and fallback provider.

    Args:
        prompt: The prompt to send
        provider: Primary provider to use
        model: Primary model to use
        api_key: API key for primary provider
        max_retries: Maximum retry attempts
        fallback_provider: Fallback provider if primary fails
        fallback_model: Fallback model if primary fails
        api_keys: Dictionary of API keys for fallback

    Returns:
        Optional[str]: LLM response or None if all attempts failed
    """
    # Resolve base URL
    primary_base_url = resolve_provider_base_url(provider, base_urls)

    # First try with primary provider
    for attempt in range(max_retries):
        try:
            if api_key:
                response = get_model_response(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=primary_base_url,
                )
                write_log(f"Successfully got response from {provider} on attempt {attempt + 1}")
                return response
            else:
                write_log(f"No API key found for {provider}, trying fallback")
                break
        except (
            requests.RequestException,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as e:
            if attempt < max_retries - 1:
                write_log(
                    f"Error on {provider} attempt {attempt + 1}/{max_retries}: {e!s}",
                    level="warning",
                )
                time.sleep(5 * (2**attempt))
            else:
                write_log(f"All {provider} retry attempts failed: {e!s}", level="warning")
                write_log(f"Falling back to {fallback_provider}")

    # Try fallback provider
    if api_keys:
        fallback_api_key = api_keys.get(fallback_provider) or load_api_key(fallback_provider)
        if fallback_api_key:
            # Resolve base URL for fallback provider
            fallback_base_url = resolve_provider_base_url(fallback_provider, base_urls)
            try:
                response = get_model_response(
                    prompt=prompt,
                    provider=fallback_provider,
                    model=fallback_model,
                    api_key=fallback_api_key,
                    base_url=fallback_base_url,
                )
                write_log(f"Successfully got response from {fallback_provider} as fallback")
                return response
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as e:
                write_log(f"Error on {fallback_provider} fallback: {e!s}", level="warning")
        else:
            write_log(f"No {fallback_provider} API key found, falling back to simple consensus")

    return None


def _extract_metrics_from_text(
    text: str,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Extract consensus metrics (CP, H) and optional annotation from text.

    This function mirrors the R implementation's parsing strategies:
    1. parse_standard_format: Try structured 4-line format [0/1, CP, H, annotation]
    2. parse_flexible_format: Try labeled patterns like "Consensus Proportion = 0.75"
    3. find_majority_prediction: Find annotation that's not a numeric pattern

    Args:
        text: Text to parse (LLM response or discussion)

    Returns:
        tuple[Optional[float], Optional[float], Optional[str]]:
            (consensus_proportion, entropy, annotation)
            annotation may be None if not found
    """
    if not text or not text.strip():
        return None, None, None

    lines = text.strip().split("\n")
    lines = [line.strip() for line in lines if line.strip()]

    if not lines:
        return None, None, None

    cp_value = None
    h_value = None
    annotation = None

    # Regex patterns (mirroring R's .CONSENSUS_CONSTANTS)
    consensus_indicator_pattern = r"^\s*[01]\s*$"
    proportion_pattern = r"^\s*(0\.\d+|1\.0*|1)\s*$"
    entropy_pattern = r"^\s*(\d+\.\d+|\d+)\s*$"
    general_numeric_pattern = r"^\s*\d+(\.\d+)?\s*$"

    # Strategy 1: Try structured 4-line format [0/1, CP, H, annotation]
    # (mirrors R's parse_standard_format) - only if we have at least 4 lines
    if len(lines) >= 4:
        result_lines = lines[-4:]
        is_line1_valid = bool(re.match(consensus_indicator_pattern, result_lines[0]))
        is_line2_valid = bool(re.match(proportion_pattern, result_lines[1]))
        is_line3_valid = bool(re.match(entropy_pattern, result_lines[2]))

        if is_line1_valid and is_line2_valid and is_line3_valid:
            write_log("Detected standard 4-line format", level="debug")
            with contextlib.suppress(ValueError, IndexError):
                cp_value = float(result_lines[1].strip())
                h_value = float(result_lines[2].strip())
                majority = result_lines[3].strip()
                annotation = majority if majority and majority != "Unknown" else None
                return cp_value, h_value, annotation

    # Strategy 2: Parse flexible format (mirrors R's parse_flexible_format)
    # This handles responses with fewer than 4 lines or non-standard formats
    write_log(f"Using flexible format parsing for {len(lines)} line(s)", level="debug")

    # Extract consensus indicator (0 or 1)
    for line in lines:
        if re.match(consensus_indicator_pattern, line):
            # Found consensus indicator, but we don't use it directly
            break

    # Extract consensus proportion from labeled lines
    # Pattern: "Consensus Proportion = 0.75" or "Consensus Proportion: 0.75"
    cp_patterns = [
        r"(?i)consensus\s+proportion\s*(?:\(CP\))?\s*[:=]\s*([0-9.]+)",
        r"(?i)CP\s*[:=]\s*([0-9.]+)",
    ]
    for line in lines:
        if "consensus" in line.lower() and "proportion" in line.lower() and "=" in line:
            parts = line.split("=")
            if len(parts) > 1:
                last_part = parts[-1].strip()
                value_match = re.search(r"(0\.\d+|1\.0*|1)", last_part)
                if value_match:
                    with contextlib.suppress(ValueError):
                        potential_cp = float(value_match.group(1))
                        if 0 <= potential_cp <= 1:
                            cp_value = potential_cp
                            write_log(f"Found CP in line: {cp_value}", level="debug")
                            break

    # If not found, try regex patterns on full text
    if cp_value is None:
        for pattern in cp_patterns:
            cp_match = re.search(pattern, text)
            if cp_match:
                with contextlib.suppress(ValueError):
                    cp_value = float(cp_match.group(1))
                    if 0 <= cp_value <= 1:
                        break
                    else:
                        cp_value = None

    # Extract entropy from labeled lines
    # Pattern: "Shannon Entropy = 0.85" or "Entropy = 0.85" or "H = 0.85"
    h_patterns = [
        r"(?i)(?:shannon\s+)?entropy\s*(?:\(H\))?\s*[:=]\s*([0-9.]+)",
        r"(?i)\bH\s*[:=]\s*([0-9.]+)",
    ]
    for line in lines:
        if "entropy" in line.lower() and "=" in line:
            parts = line.split("=")
            if len(parts) > 1:
                last_part = parts[-1].strip()
                value_match = re.search(r"(\d+\.\d+|\d+)", last_part)
                if value_match:
                    with contextlib.suppress(ValueError):
                        potential_h = float(value_match.group(1))
                        if potential_h >= 0:
                            h_value = potential_h
                            write_log(f"Found H in line: {h_value}", level="debug")
                            break

    # If not found, try regex patterns on full text
    if h_value is None:
        for pattern in h_patterns:
            h_match = re.search(pattern, text)
            if h_match:
                with contextlib.suppress(ValueError):
                    h_value = float(h_match.group(1))
                    if h_value >= 0:
                        break
                    else:
                        h_value = None

    # Find majority prediction (mirrors R's find_majority_prediction)
    # Skip lines that match numeric patterns or contain labels
    label_patterns = ["consensus", "proportion", "entropy"]
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Skip numeric lines
        if re.match(general_numeric_pattern, line_clean):
            continue
        if re.match(consensus_indicator_pattern, line_clean):
            continue
        if re.match(proportion_pattern, line_clean):
            continue
        if re.match(entropy_pattern, line_clean):
            continue

        # Skip lines containing labels
        if any(label in line_clean.lower() for label in label_patterns):
            continue

        # This line is likely the majority prediction
        annotation = line_clean
        break

    return cp_value, h_value, annotation


def check_consensus(
    predictions: dict[str, dict[str, str]],
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    api_keys: Optional[dict[str, str]] = None,
    return_controversial: bool = True,
    consensus_model: Optional[dict[str, str]] = None,
    available_models: Optional[list[Union[str, dict[str, str]]]] = None,
) -> Union[
    tuple[dict[str, str], dict[str, float], dict[str, float]],
    tuple[dict[str, str], dict[str, float], dict[str, float], list[str]],
]:
    """Check consensus among different model predictions using LLM assistance.

    This function uses an LLM to evaluate semantic similarity between
    annotations and optionally identifies controversial clusters.

    Args:
        predictions: Dictionary mapping model names to dictionaries of cluster annotations
        consensus_threshold: Agreement threshold below which a cluster is considered controversial
        entropy_threshold: Entropy threshold above which a cluster is considered controversial
        api_keys: Dictionary mapping provider names to API keys
        return_controversial: Whether to return controversial clusters list
        consensus_model: Optional dict with 'provider' and 'model' keys to specify which model
            to use for consensus checking. If not provided, will try available_models first,
            then defaults to Qwen with Anthropic fallback.
        available_models: Optional list of models that are available for use. Used as fallback
            when consensus_model is not specified and default models are not available.

    Returns:
        Tuple of:
            - Dictionary mapping cluster IDs to consensus annotations
            - Dictionary mapping cluster IDs to consensus proportion scores
            - Dictionary mapping cluster IDs to entropy scores
            - List of controversial cluster IDs (only if return_controversial=True)
    """
    consensus = {}
    consensus_proportion = {}
    entropy = {}

    # Filter out models with empty results, but keep models with valid annotations
    # This allows consensus calculation even if some models failed
    valid_predictions = {
        model: results
        for model, results in predictions.items()
        if results  # Keep only models that returned non-empty results
    }

    # Check if we have enough valid predictions
    if not valid_predictions:
        write_log("No valid predictions from any model", level="warning")
        if return_controversial:
            return {}, {}, {}, []
        return {}, {}, {}

    if len(valid_predictions) < len(predictions):
        failed_models = set(predictions.keys()) - set(valid_predictions.keys())
        write_log(
            f"Some models returned empty results, excluding: {failed_models}",
            level="warning",
        )

    # Use filtered predictions for the rest of the function
    predictions = valid_predictions

    # Get all clusters
    all_clusters = set()
    for model_results in predictions.values():
        all_clusters.update(model_results.keys())

    # Process each cluster
    for cluster in all_clusters:
        # Collect all annotations for this cluster
        cluster_annotations = []

        for _model, results in predictions.items():
            if cluster in results:
                annotation = clean_annotation(results[cluster])
                if annotation:
                    cluster_annotations.append(annotation)

        if len(cluster_annotations) < 2:
            # Not enough annotations to check consensus
            if cluster_annotations:
                consensus[cluster] = cluster_annotations[0]
                consensus_proportion[cluster] = 1.0
                entropy[cluster] = 0.0
            else:
                consensus[cluster] = "Unknown"
                consensus_proportion[cluster] = 0.0
                entropy[cluster] = 0.0
            continue

        # OPTIMIZATION: First try simple consensus calculation
        write_log(f"Starting with simple consensus calculation for cluster {cluster}", level="info")

        # Use unified consensus calculation
        simple_cp, simple_entropy, majority_annotation = _calculate_simple_consensus(
            cluster_annotations
        )

        # Check if simple consensus meets thresholds
        if simple_cp >= consensus_threshold and simple_entropy <= entropy_threshold:
            # Simple consensus is sufficient
            consensus_proportion[cluster] = simple_cp
            entropy[cluster] = simple_entropy
            consensus[cluster] = majority_annotation
            write_log(
                f"Cluster {cluster} achieved consensus with simple check: "
                f"CP={simple_cp:.2f}, H={simple_entropy:.2f}",
                level="info",
            )
            continue

        # Simple consensus didn't meet thresholds, use LLM for double-checking
        write_log(
            f"Cluster {cluster} needs LLM double-check: CP={simple_cp:.2f}, H={simple_entropy:.2f}",
            level="info",
        )

        # Create prompt for LLM
        prompt = create_consensus_check_prompt(cluster_annotations)

        # Determine which model to use
        if consensus_model:
            primary_provider = consensus_model.get("provider", "qwen")
            primary_model = consensus_model.get("model", "qwen-max-2025-01-25")
        else:
            # Default to Qwen if not specified
            primary_provider = "qwen"
            primary_model = "qwen-max-2025-01-25"

        # Get API key for primary provider
        primary_api_key = (api_keys.get(primary_provider) if api_keys else None) or load_api_key(
            primary_provider
        )

        # If primary model is not available and we have available_models, try to use one of them
        if not primary_api_key and available_models and api_keys and not consensus_model:
            write_log(
                f"Primary consensus model {primary_provider} not available, trying available models",
                level="info",
            )

            # Try to find a suitable model from available_models
            for model_item in available_models:
                if isinstance(model_item, dict):
                    alt_provider = model_item.get("provider")
                    alt_model = model_item.get("model")
                    if not alt_provider and alt_model:
                        alt_provider = get_provider(alt_model)
                else:
                    alt_model = model_item
                    alt_provider = get_provider(model_item)

                # Check if we have API key for this alternative model
                if alt_provider and alt_provider in api_keys:
                    primary_provider = alt_provider
                    primary_model = alt_model
                    primary_api_key = api_keys[alt_provider]
                    write_log(
                        f"Using available model {alt_model} from {alt_provider} for consensus checking",
                        level="info",
                    )
                    break

        # Call LLM with retry and fallback logic
        llm_response = _call_llm_with_retry(
            prompt=prompt,
            provider=primary_provider,
            model=primary_model,
            api_key=primary_api_key,
            max_retries=3,
            fallback_provider=DEFAULT_FALLBACK_PROVIDER,
            fallback_model=DEFAULT_FALLBACK_MODEL,
            api_keys=api_keys,
        )

        # Parse LLM response using unified parser
        if llm_response:
            llm_cp, llm_entropy, llm_prediction = _extract_metrics_from_text(llm_response)

            if llm_cp is not None and llm_entropy is not None:
                consensus_proportion[cluster] = llm_cp
                entropy[cluster] = llm_entropy
                # Use LLM's annotation if available, otherwise use simple consensus result
                consensus[cluster] = llm_prediction or majority_annotation
                write_log(
                    f"LLM consensus check for cluster {cluster}: "
                    f"CP={llm_cp:.2f}, H={llm_entropy:.2f}",
                    level="info",
                )
                continue

        # If LLM failed to provide metrics, use simple consensus results
        consensus_proportion[cluster] = simple_cp
        entropy[cluster] = simple_entropy
        consensus[cluster] = majority_annotation
        write_log(
            f"Using simple consensus for cluster {cluster} after LLM failure: "
            f"CP={simple_cp:.2f}, H={simple_entropy:.2f}",
            level="info",
        )

    if return_controversial:
        # Find controversial clusters based on both consensus proportion and entropy
        controversial = [
            cluster
            for cluster, score in consensus_proportion.items()
            if score < consensus_threshold or entropy.get(cluster, 0) > entropy_threshold
        ]
        return consensus, consensus_proportion, entropy, controversial

    return consensus, consensus_proportion, entropy


def _extract_cell_type_from_response(response: str) -> Optional[str]:
    """Extract cell type from a discussion response.

    This function mirrors the R implementation's logic:
    - For multi-line responses (discussion format): Looks for 'CELL TYPE:' pattern
    - For single-line responses: Returns the response as-is (it's likely already
      a cell type annotation)

    Args:
        response: The model's discussion response

    Returns:
        Optional[str]: Extracted cell type or None
    """
    if not response:
        return None

    response = response.strip()
    if not response:
        return None

    # Check if this is a multi-line response (discussion format)
    lines = response.split("\n")
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    if len(non_empty_lines) > 1:
        # Multi-line response - look for CELL TYPE: pattern
        for line in non_empty_lines:
            # Match lines starting with "CELL TYPE:" (case-insensitive)
            if re.match(r"^\s*CELL\s*TYPE\s*:", line, re.IGNORECASE):
                # Extract the cell type after the colon
                cell_type = re.sub(r"^\s*CELL\s*TYPE\s*:\s*", "", line, flags=re.IGNORECASE)
                cell_type = cell_type.strip()
                # Clean up common artifacts like [bracketed text]
                cell_type = re.sub(r"\[.*?\]", "", cell_type).strip()
                if cell_type and cell_type.lower() not in ["unknown", "unclear", "n/a"]:
                    return cell_type
        # If no CELL TYPE: found in multi-line, return None
        return None
    else:
        # Single-line response - return as is (it's likely already a cell type)
        cell_type = non_empty_lines[0] if non_empty_lines else None
        if cell_type:
            # Clean up if it has a colon (might be "cluster_id: cell_type" format)
            if ":" in cell_type:
                parts = cell_type.split(":", 1)
                # Check if the first part looks like a cluster ID or "CELL TYPE"
                first_part = parts[0].strip().lower()
                if first_part.isdigit() or first_part in ["cell type", "celltype"]:
                    cell_type = parts[1].strip()
            # Clean up common artifacts
            cell_type = re.sub(r"\[.*?\]", "", cell_type).strip()
            if cell_type and cell_type.lower() not in ["unknown", "unclear", "n/a"]:
                return cell_type
        return None


def _calculate_simple_consensus(
    annotations: list[str],
) -> tuple[float, float, str]:
    """Calculate simple consensus metrics from a list of annotations.

    This is the core consensus calculation logic used by both check_consensus
    and check_consensus_for_discussion_round. It implements:
    1. Normalize annotations for fair comparison
    2. Count occurrences of each normalized annotation
    3. Calculate consensus proportion (CP) and Shannon entropy (H)
    4. Return the majority prediction (original form)

    Args:
        annotations: List of cell type annotations to analyze

    Returns:
        tuple[float, float, str]: (consensus_proportion, entropy, majority_prediction)
    """
    if not annotations:
        return 0.0, DEFAULT_FALLBACK_ENTROPY, "Unknown"

    # Normalize annotations and group originals
    normalized_groups: dict[str, list[str]] = {}
    for original in annotations:
        normalized = normalize_annotation_for_comparison(original)
        if normalized not in normalized_groups:
            normalized_groups[normalized] = []
        normalized_groups[normalized].append(original)

    # Count by normalized groups
    normalized_counts = {norm: len(originals) for norm, originals in normalized_groups.items()}
    total = len(annotations)

    # Find most common normalized group
    if not normalized_counts:
        return 0.0, DEFAULT_FALLBACK_ENTROPY, "Unknown"

    most_common_norm = max(normalized_counts.items(), key=lambda x: x[1])
    most_common_norm_key = most_common_norm[0]
    most_common_count = most_common_norm[1]

    # Get the most frequent original annotation from the majority normalized group
    original_annotations = normalized_groups[most_common_norm_key]
    original_counts = Counter(original_annotations)
    majority_prediction = original_counts.most_common(1)[0][0]

    # Calculate consensus proportion
    consensus_proportion = most_common_count / total

    # Calculate Shannon entropy with epsilon to avoid log(0)
    # Consistent with R implementation: log2(p + 1e-10)
    entropy = 0.0
    for count in normalized_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p + 1e-10)

    return consensus_proportion, entropy, majority_prediction


def check_consensus_for_discussion_round(
    round_responses: dict[str, str],
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    api_keys: Optional[dict[str, str]] = None,
    consensus_check_model: Optional[dict[str, str]] = None,
    base_urls: Optional[Union[str, dict[str, str]]] = None,
) -> dict[str, Any]:
    """Check consensus among model responses for a single discussion round.

    This function mirrors the R implementation's check_consensus function:
    1. First tries simple consensus calculation
    2. If simple consensus meets thresholds, returns immediately
    3. If not, uses LLM for double-checking
    4. Falls back to simple consensus if LLM fails

    Args:
        round_responses: Dictionary mapping model names to their responses
        consensus_threshold: Agreement threshold (default: 0.7)
        entropy_threshold: Entropy threshold (default: 1.0)
        api_keys: Dictionary mapping provider names to API keys
        consensus_check_model: Optional dict with 'provider' and 'model' keys
        base_urls: Custom base URLs for API endpoints

    Returns:
        dict containing:
            - reached: bool, whether consensus was reached
            - consensus_proportion: float, the consensus proportion
            - entropy: float, the Shannon entropy
            - majority_prediction: str, the majority cell type prediction
    """
    # Validate input - need at least 2 responses
    if len(round_responses) < 2:
        write_log(
            f"Not enough responses to check consensus: {len(round_responses)}",
            level="warning",
        )
        if len(round_responses) == 1:
            # Extract cell type from the single response
            single_response = list(round_responses.values())[0]
            cell_type = _extract_cell_type_from_response(single_response)
            return {
                "reached": False,
                "consensus_proportion": 1.0,
                "entropy": 0.0,
                "majority_prediction": cell_type or "Unknown",
            }
        return DEFAULT_CONSENSUS_RESULT.copy()

    # Extract cell types from responses
    extracted_cell_types = {}
    for model_name, response in round_responses.items():
        cell_type = _extract_cell_type_from_response(response)
        extracted_cell_types[model_name] = cell_type if cell_type else "Unknown"

    write_log(f"Extracted cell types: {extracted_cell_types}", level="debug")

    # Calculate simple consensus using unified function
    cell_type_list = list(extracted_cell_types.values())
    simple_cp, simple_entropy, majority_prediction = _calculate_simple_consensus(cell_type_list)

    write_log(
        f"Simple consensus: CP={simple_cp:.2f}, H={simple_entropy:.2f}, "
        f"majority={majority_prediction}",
        level="info",
    )

    # Check if simple consensus meets thresholds
    if simple_cp >= consensus_threshold and simple_entropy <= entropy_threshold:
        # Simple consensus is sufficient - no LLM needed
        write_log(
            "Consensus achieved with simple check - NO LLM NEEDED",
            level="info",
        )
        return {
            "reached": True,
            "consensus_proportion": simple_cp,
            "entropy": simple_entropy,
            "majority_prediction": majority_prediction,
        }

    # Simple consensus didn't meet thresholds, use LLM for double-checking
    write_log(
        f"Simple consensus below threshold (CP={simple_cp:.2f} < {consensus_threshold} OR "
        f"H={simple_entropy:.2f} > {entropy_threshold}), using LLM double-check",
        level="info",
    )

    # Create prompt for LLM consensus check
    prompt = create_consensus_check_prompt(cell_type_list)

    # Determine which model to use for consensus checking
    if consensus_check_model:
        primary_provider = consensus_check_model.get("provider", DEFAULT_FALLBACK_PROVIDER)
        primary_model = consensus_check_model.get("model", DEFAULT_FALLBACK_MODEL)
    else:
        primary_provider = DEFAULT_FALLBACK_PROVIDER
        primary_model = DEFAULT_FALLBACK_MODEL

    # Get API key
    if api_keys is None:
        api_keys = {}
    primary_api_key = api_keys.get(primary_provider) or load_api_key(primary_provider)

    # Call LLM with retry logic
    llm_response = _call_llm_with_retry(
        prompt=prompt,
        provider=primary_provider,
        model=primary_model,
        api_key=primary_api_key,
        max_retries=3,
        fallback_provider=DEFAULT_FALLBACK_PROVIDER,
        fallback_model=DEFAULT_FALLBACK_MODEL,
        api_keys=api_keys,
        base_urls=base_urls,
    )

    # Parse LLM response
    if llm_response:
        cp_value, entropy_value, llm_prediction = _extract_metrics_from_text(llm_response)

        if cp_value is not None and entropy_value is not None:
            # Use LLM's metrics
            reached = cp_value >= consensus_threshold and entropy_value <= entropy_threshold

            write_log(
                f"LLM consensus check: CP={cp_value:.2f}, H={entropy_value:.2f}, reached={reached}",
                level="info",
            )

            return {
                "reached": reached,
                "consensus_proportion": cp_value,
                "entropy": entropy_value,
                "majority_prediction": llm_prediction or majority_prediction,
            }

    # LLM failed - fall back to simple consensus results
    write_log(
        "LLM consensus check failed, using simple consensus results",
        level="warning",
    )

    reached = simple_cp >= consensus_threshold and simple_entropy <= entropy_threshold
    return {
        "reached": reached,
        "consensus_proportion": simple_cp,
        "entropy": simple_entropy,
        "majority_prediction": majority_prediction,
    }


def process_controversial_clusters(
    marker_genes: dict[str, list[str]],
    controversial_clusters: list[str],
    model_predictions: dict[str, dict[str, str]],
    species: str,
    tissue: Optional[str] = None,
    models: Optional[list[Union[str, dict[str, str]]]] = None,
    api_keys: Optional[dict[str, str]] = None,
    max_discussion_rounds: int = 3,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    base_urls: Optional[Union[str, dict[str, str]]] = None,
    force_rerun: bool = False,
    consensus_check_model: Optional[dict[str, str]] = None,
) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, float], dict[str, float]]:
    """Process controversial clusters through multi-model discussion.

    This function facilitates a real discussion between multiple LLMs, where each
    model can see and respond to other models' arguments. This mirrors the R
    implementation where all models participate in each round of discussion.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        controversial_clusters: List of controversial cluster IDs
        model_predictions: Dictionary mapping model names to dictionaries of
            cluster annotations (from initial annotation phase)
        species: Species name (e.g., 'human', 'mouse')
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        models: List of models to participate in the discussion. Each item can be
            a string (model name) or dict with 'provider' and 'model' keys.
        api_keys: Dictionary mapping provider names to API keys
        max_discussion_rounds: Maximum number of discussion rounds
        consensus_threshold: Consensus proportion threshold for agreement
        entropy_threshold: Entropy threshold for agreement
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        base_urls: Custom base URLs for API endpoints
        force_rerun: If True, ignore cached results
        consensus_check_model: Optional dict with 'provider' and 'model' keys
            to specify which model to use for consensus checking with LLM.
            If not provided, uses default fallback model.

    Returns:
        tuple containing:
            - results: Dict mapping cluster IDs to resolved annotations
            - discussion_history: Dict mapping cluster IDs to list of round responses
            - updated_consensus_proportion: Dict mapping cluster IDs to CP scores
            - updated_entropy: Dict mapping cluster IDs to entropy scores
    """
    if not models:
        write_log("No models provided for discussion", level="error")
        return {}, {}, {}, {}

    if not api_keys:
        api_keys = {}

    results = {}
    discussion_history = {}
    updated_consensus_proportion = {}
    updated_entropy = {}

    # Prepare model information
    model_info_list = []
    for model_item in models:
        if isinstance(model_item, dict):
            provider = model_item.get("provider")
            model_name = model_item.get("model")
            if not provider and model_name:
                provider = get_provider(model_name)
        else:
            model_name = model_item
            provider = get_provider(model_name)

        api_key = api_keys.get(provider) or load_api_key(provider)
        if not api_key:
            write_log(f"No API key for {provider}, skipping {model_name}", level="warning")
            continue

        base_url = resolve_provider_base_url(provider, base_urls)
        model_info_list.append(
            {
                "name": model_name,
                "provider": provider,
                "api_key": api_key,
                "base_url": base_url,
            }
        )

    if len(model_info_list) < 2:
        write_log(
            f"Need at least 2 models for discussion, only have {len(model_info_list)}",
            level="error",
        )
        return {}, {}, {}, {}

    write_log(f"Starting multi-model discussion with {len(model_info_list)} models")

    for cluster_id in controversial_clusters:
        write_log(f"Processing controversial cluster {cluster_id}")

        # Get marker genes for this cluster
        current_marker_genes = marker_genes.get(cluster_id, [])
        if not current_marker_genes:
            write_log(f"No marker genes found for cluster {cluster_id}", level="warning")
            results[cluster_id] = "Unknown (no markers)"
            discussion_history[cluster_id] = []
            # Ensure CP/H are populated for consistency
            updated_consensus_proportion[cluster_id] = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
            updated_entropy[cluster_id] = DEFAULT_FALLBACK_ENTROPY
            continue

        # Get initial predictions for this cluster
        initial_predictions = {
            model_name: predictions.get(cluster_id, "Unknown")
            for model_name, predictions in model_predictions.items()
            if cluster_id in predictions
        }

        # Initialize discussion tracking
        rounds_history = []  # List of dicts: [{model: response, ...}, ...]
        final_decision = None
        current_cp = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
        current_h = DEFAULT_FALLBACK_ENTROPY

        try:
            for current_round in range(1, max_discussion_rounds + 1):
                write_log(f"Starting round {current_round} for cluster {cluster_id}")

                round_responses = {}

                # Generate prompt for this round
                if current_round == 1:
                    # Initial round - each model sees the initial predictions
                    prompt = create_initial_discussion_prompt(
                        cluster_id=cluster_id,
                        marker_genes=current_marker_genes,
                        initial_predictions=initial_predictions,
                        species=species,
                        tissue=tissue,
                    )
                else:
                    # Subsequent rounds - each model sees all previous discussions
                    prompt = create_discussion_prompt(
                        cluster_id=cluster_id,
                        marker_genes=current_marker_genes,
                        previous_rounds=rounds_history,
                        round_number=current_round,
                        species=species,
                        tissue=tissue,
                    )

                # Get response from each model
                for model_info in model_info_list:
                    model_name = model_info["name"]
                    try:
                        response = get_model_response(
                            prompt=prompt,
                            provider=model_info["provider"],
                            model=model_name,
                            api_key=model_info["api_key"],
                            use_cache=use_cache and not force_rerun,
                            cache_dir=cache_dir,
                            base_url=model_info["base_url"],
                        )
                        round_responses[model_name] = response
                        write_log(f"Got response from {model_name} in round {current_round}")
                    except Exception as e:
                        write_log(
                            f"Error getting response from {model_name}: {e!s}",
                            level="warning",
                        )
                        round_responses[model_name] = f"Error: {e!s}"

                # Store this round's responses
                rounds_history.append(round_responses)

                # Filter out error responses for consensus check
                valid_responses = {
                    k: v for k, v in round_responses.items() if not v.startswith("Error:")
                }

                if len(valid_responses) < 2:
                    write_log(
                        f"Only {len(valid_responses)} valid responses in round {current_round}",
                        level="warning",
                    )
                    continue

                # Check consensus using LLM double-check (mirrors R implementation)
                consensus_result = check_consensus_for_discussion_round(
                    round_responses=valid_responses,
                    consensus_threshold=consensus_threshold,
                    entropy_threshold=entropy_threshold,
                    api_keys=api_keys,
                    consensus_check_model=consensus_check_model,
                    base_urls=base_urls,
                )

                current_cp = consensus_result["consensus_proportion"]
                current_h = consensus_result["entropy"]
                majority = consensus_result["majority_prediction"]

                write_log(
                    f"Round {current_round} consensus: CP={current_cp:.2f}, H={current_h:.2f}, "
                    f"majority={majority}, reached={consensus_result['reached']}",
                    level="info",
                )

                # Check if consensus is reached (using the result from check_consensus_for_discussion_round)
                if consensus_result["reached"]:
                    final_decision = majority
                    write_log(
                        f"Consensus reached in round {current_round} for cluster {cluster_id}: {final_decision}",
                        level="info",
                    )
                    break

            # After all rounds, determine final result
            if not final_decision and rounds_history:
                # Use the majority from the last round with LLM verification
                last_round = rounds_history[-1]
                valid_last = {k: v for k, v in last_round.items() if not v.startswith("Error:")}
                if valid_last:
                    last_consensus = check_consensus_for_discussion_round(
                        round_responses=valid_last,
                        consensus_threshold=consensus_threshold,
                        entropy_threshold=entropy_threshold,
                        api_keys=api_keys,
                        consensus_check_model=consensus_check_model,
                        base_urls=base_urls,
                    )
                    final_decision = last_consensus["majority_prediction"]
                    current_cp = last_consensus["consensus_proportion"]
                    current_h = last_consensus["entropy"]
                    write_log(
                        f"Using last round majority for cluster {cluster_id}: {final_decision} "
                        f"(CP={current_cp:.2f}, H={current_h:.2f})",
                        level="info",
                    )

            # Store results
            if final_decision and final_decision != "Unknown":
                results[cluster_id] = clean_annotation(final_decision)
                updated_consensus_proportion[cluster_id] = current_cp
                updated_entropy[cluster_id] = current_h
            else:
                results[cluster_id] = "Inconclusive"
                updated_consensus_proportion[cluster_id] = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
                updated_entropy[cluster_id] = DEFAULT_FALLBACK_ENTROPY

            discussion_history[cluster_id] = rounds_history

        except Exception as e:
            write_log(f"Error during discussion for cluster {cluster_id}: {e!s}", level="error")
            results[cluster_id] = f"Error: {e!s}"
            discussion_history[cluster_id] = []
            updated_consensus_proportion[cluster_id] = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
            updated_entropy[cluster_id] = DEFAULT_FALLBACK_ENTROPY

    return results, discussion_history, updated_consensus_proportion, updated_entropy


def interactive_consensus_annotation(
    marker_genes: dict[str, list[str]],
    species: str,
    models: list[Union[str, dict[str, str]]] = None,
    api_keys: Optional[dict[str, str]] = None,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    max_discussion_rounds: int = 3,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
    consensus_model: Optional[Union[str, dict[str, str]]] = None,
    base_urls: Optional[Union[str, dict[str, str]]] = None,
    clusters_to_analyze: Optional[list[str]] = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Perform consensus annotation of cell types using multiple LLMs and interactive resolution.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        models: List of models to use for annotation
        api_keys: Dictionary mapping provider names to API keys
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        consensus_threshold: Agreement threshold below which a cluster is considered controversial
        entropy_threshold: Entropy threshold above which a cluster is considered controversial
        max_discussion_rounds: Maximum number of discussion rounds for controversial clusters
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        verbose: Whether to print detailed logs
        consensus_model: Optional model specification for consensus checking and discussion.
            Can be a string (model name) or dict with 'provider' and 'model' keys.
            If not provided, defaults to Qwen for consensus checking and selects from
            input models for discussion.
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs
        clusters_to_analyze: Optional list of cluster IDs to analyze. If provided,
            only the specified clusters will be processed. Cluster IDs must exist
            in the marker_genes dictionary. Non-existent cluster IDs will be
            ignored with a warning. If None (default), all clusters will be analyzed.
        force_rerun: If True, ignore cached results and force re-analysis of all
            specified clusters. Useful when you want to re-analyze clusters with
            different context or for subtype identification. Default is False.
            Note: This parameter only affects the discussion phase for controversial
            clusters when use_cache is True.

    Returns:
        dict[str, Any]: Dictionary containing consensus results and metadata

    """
    # Set up logging
    if verbose:
        write_log("Starting interactive consensus annotation")

    # Filter clusters if clusters_to_analyze is specified
    if clusters_to_analyze is not None:
        # Convert to list of strings for consistent comparison
        clusters_to_analyze = [str(cluster_id) for cluster_id in clusters_to_analyze]

        # Get all available clusters
        available_clusters = list(marker_genes.keys())

        # Check which requested clusters exist
        valid_clusters = [
            cluster_id for cluster_id in clusters_to_analyze if cluster_id in available_clusters
        ]
        invalid_clusters = [
            cluster_id for cluster_id in clusters_to_analyze if cluster_id not in available_clusters
        ]

        # Warn about non-existent clusters
        if invalid_clusters:
            warning_msg = f"The following cluster IDs were not found in the input: {', '.join(invalid_clusters)}"
            write_log(warning_msg, level="warning")
            if verbose:
                print(f"Warning: {warning_msg}")

        # Stop if no valid clusters
        if not valid_clusters:
            error_msg = "None of the specified clusters exist in the input data."
            write_log(error_msg, level="error")
            raise ValueError(error_msg)

        # Filter marker_genes to only include specified clusters
        marker_genes = {cluster_id: marker_genes[cluster_id] for cluster_id in valid_clusters}

        # Log the filtering
        log_msg = f"Filtered to analyze {len(valid_clusters)} clusters: {', '.join(valid_clusters)}"
        write_log(log_msg)
        if verbose:
            print(
                f"Info: Analyzing {len(valid_clusters)} specified clusters: {', '.join(valid_clusters)}"
            )

    # Make sure we have API keys
    if api_keys is None:
        api_keys = {}
        for model_item in models:
            # Handle both string models and dict models
            if isinstance(model_item, dict):
                provider = model_item.get("provider")
                if not provider:
                    # Try to get provider from model name if not explicitly provided
                    provider = get_provider(model_item.get("model", ""))
            else:
                provider = get_provider(model_item)

            if provider and provider not in api_keys:
                api_key = load_api_key(provider)
                if api_key:
                    api_keys[provider] = api_key

    # Process consensus_model parameter
    consensus_model_dict = None
    if consensus_model:
        if isinstance(consensus_model, str):
            # If it's a string, get the provider
            consensus_provider = get_provider(consensus_model)
            consensus_model_dict = {"provider": consensus_provider, "model": consensus_model}
        else:
            # It's already a dict
            consensus_model_dict = consensus_model

        # Ensure we have API key for consensus model
        consensus_provider = consensus_model_dict.get("provider")
        if consensus_provider and consensus_provider not in api_keys:
            api_key = load_api_key(consensus_provider)
            if api_key:
                api_keys[consensus_provider] = api_key

    # Run initial annotations with all models
    model_results = {}

    for model_item in models:
        # Handle both string models and dict models
        if isinstance(model_item, dict):
            provider = model_item.get("provider")
            model_name = model_item.get("model")

            # If provider is not explicitly provided, try to get it from model name
            if not provider:
                provider = get_provider(model_name)
        else:
            provider = get_provider(model_item)
            model_name = model_item

        api_key = api_keys.get(provider)

        # For OpenRouter models, we need to keep the full model name with the provider prefix
        # The model name is already in the correct format (e.g., "openai/gpt-5")
        # Do not modify the model name for OpenRouter

        if not api_key:
            write_log(
                f"Warning: No API key found for {provider}, skipping {model_name}",
                level="warning",
            )
            continue

        if verbose:
            write_log(f"Annotating with {model_name}")

        try:
            results = annotate_clusters(
                marker_genes=marker_genes,
                species=species,
                provider=provider,
                model=model_name,
                api_key=api_key,
                tissue=tissue,
                additional_context=additional_context,
                use_cache=use_cache and not force_rerun,
                cache_dir=cache_dir,
                base_urls=base_urls,
            )

            model_results[model_name] = results

            if verbose:
                write_log(f"Successfully annotated with {model_name}")
        except (
            requests.RequestException,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            AttributeError,
            ImportError,
        ) as e:
            write_log(f"Error annotating with {model_name}: {e!s}", level="error")

    # Check if we have any results
    if not model_results:
        write_log("No annotations were successful", level="error")
        return {"error": "No annotations were successful"}

    # Check consensus
    consensus, consensus_proportion, entropy, controversial = check_consensus(
        model_results,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        api_keys=api_keys,
        consensus_model=consensus_model_dict,
        available_models=models,
    )

    if verbose:
        write_log(f"Found {len(controversial)} controversial clusters out of {len(consensus)}")

    # If there are controversial clusters, resolve them through multi-model discussion
    resolved = {}
    discussion_logs = {}
    if controversial:
        if verbose:
            write_log(
                f"Resolving {len(controversial)} controversial clusters through multi-model discussion"
            )

        try:
            resolved, discussion_logs, updated_cp, updated_h = process_controversial_clusters(
                marker_genes=marker_genes,
                controversial_clusters=controversial,
                model_predictions=model_results,
                species=species,
                tissue=tissue,
                models=models,  # Pass all models for multi-model discussion
                api_keys=api_keys,
                max_discussion_rounds=max_discussion_rounds,
                consensus_threshold=consensus_threshold,
                entropy_threshold=entropy_threshold,
                use_cache=use_cache,
                cache_dir=cache_dir,
                base_urls=base_urls,
                force_rerun=force_rerun,
                consensus_check_model=consensus_model_dict,  # Pass consensus model for LLM verification
            )

            # Update consensus proportion and entropy for resolved clusters
            for cluster_id, cp_value in updated_cp.items():
                consensus_proportion[cluster_id] = cp_value

            for cluster_id, h_value in updated_h.items():
                entropy[cluster_id] = h_value

            if verbose:
                write_log(f"Successfully resolved {len(resolved)} controversial clusters")
        except (
            requests.RequestException,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            AttributeError,
        ) as e:
            write_log(f"Error resolving controversial clusters: {e!s}", level="error")

    # Merge consensus and resolved
    final_annotations = consensus.copy()
    for cluster_id, annotation in resolved.items():
        final_annotations[cluster_id] = annotation

    # Clean all annotations, ensure special markers are removed
    cleaned_annotations = {}
    for cluster_id, annotation in final_annotations.items():
        cleaned_annotations[cluster_id] = clean_annotation(annotation)

    # Prepare results
    return {
        "consensus": cleaned_annotations,
        "consensus_proportion": consensus_proportion,
        "entropy": entropy,
        "controversial_clusters": controversial,
        "resolved": resolved,
        "model_annotations": model_results,
        "discussion_logs": discussion_logs,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": models,
            "species": species,
            "tissue": tissue,
            "consensus_threshold": consensus_threshold,
            "entropy_threshold": entropy_threshold,
            "max_discussion_rounds": max_discussion_rounds,
        },
    }
