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

from .annotate import get_model_response
from .logger import write_log
from .prompts import create_discussion_consensus_check_prompt, create_discussion_prompt
from .utils import clean_annotation, normalize_annotation_for_comparison


def _get_api_key(provider: str, api_keys: Optional[dict[str, str]] = None) -> Optional[str]:
    """Get API key for a specific provider.

    Args:
        provider: Provider name (e.g., 'qwen', 'anthropic')
        api_keys: Optional dictionary of API keys

    Returns:
        Optional[str]: API key if found, None otherwise
    """
    # Try to get from provided api_keys first
    if api_keys and provider in api_keys:
        return api_keys[provider]

    # Fallback to loading from environment/config
    from .utils import load_api_key

    return load_api_key(provider)


def _handle_llm_error(
    error: Exception, context: str, attempt: int = 0, max_attempts: int = 1
) -> None:
    """Handle LLM API call errors consistently.

    Args:
        error: The exception that occurred
        context: Context description (e.g., "Qwen attempt", "Claude fallback")
        attempt: Current attempt number (0-based)
        max_attempts: Maximum number of attempts
    """
    if attempt < max_attempts - 1:
        write_log(f"Error on {context} {attempt + 1}: {str(error)}", level="warning")
        write_log("Waiting before next attempt...")
    else:
        write_log(f"Error on {context}: {str(error)}", level="warning")
        if "attempt" in context.lower():
            write_log(f"All {context.split()[0]} retry attempts failed")


def _call_llm_with_retry(
    prompt: str,
    provider: str,
    model: str,
    api_key: Optional[str],
    max_retries: int = 3,
    fallback_provider: str = "anthropic",
    fallback_model: str = "claude-3-5-sonnet-latest",
    api_keys: Optional[dict[str, str]] = None,
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
    # First try with primary provider
    for attempt in range(max_retries):
        try:
            if api_key:
                response = get_model_response(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    api_key=api_key,
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
            _handle_llm_error(e, f"{provider} attempt", attempt, max_retries)
            if attempt < max_retries - 1:
                time.sleep(5 * (2**attempt))
            else:
                write_log(f"falling back to {fallback_provider}")

    # Try fallback provider
    if api_keys:
        fallback_api_key = _get_api_key(fallback_provider, api_keys)
        if fallback_api_key:
            try:
                response = get_model_response(
                    prompt=prompt,
                    provider=fallback_provider,
                    model=fallback_model,
                    api_key=fallback_api_key,
                )
                write_log(f"Successfully got response from {fallback_provider} as fallback")
                return response
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as e:
                _handle_llm_error(e, f"{fallback_provider} fallback")
        else:
            write_log(f"No {fallback_provider} API key found, falling back to simple consensus")

    return None


def _parse_llm_consensus_response(
    response: str,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse LLM consensus response to extract metrics and annotation.

    Args:
        response: LLM response text

    Returns:
        tuple[Optional[float], Optional[float], Optional[str]]: (consensus_proportion, entropy, annotation)
    """
    try:
        # Split response by newlines and clean up
        lines = response.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        # Get the last 4 non-empty lines (standard format)
        if len(lines) >= 4:
            result_lines = lines[-4:]

            # Check if it's a standard format (0/1, proportion, entropy, annotation)
            if (
                re.match(r"^\s*[01]\s*$", result_lines[0])
                and re.match(r"^\s*(0\.\d+|1\.0*|1)\s*$", result_lines[1])
                and re.match(r"^\s*(\d+\.\d+|\d+)\s*$", result_lines[2])
            ):
                # Extract consensus proportion
                prop_value = float(result_lines[1].strip())

                # Extract entropy value
                entropy_value = float(result_lines[2].strip())

                # Extract majority prediction
                majority_prediction = result_lines[3].strip()
                annotation = (
                    majority_prediction
                    if majority_prediction and majority_prediction != "Unknown"
                    else "Unknown"
                )

                return prop_value, entropy_value, annotation
    except (ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
        write_log(f"Error parsing LLM response: {str(e)}", level="warning")

    return None, None, None


def check_consensus(
    predictions: dict[str, dict[str, str]],
    consensus_threshold: float = 0.6,
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
    from .prompts import create_consensus_check_prompt

    consensus = {}
    consensus_proportion = {}
    entropy = {}

    # Ensure we have annotations
    if not predictions or not all(predictions.values()):
        if return_controversial:
            return {}, {}, {}, []
        return {}, {}, {}

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

        # Normalize annotations for better comparison
        normalized_annotations = {}
        for original in cluster_annotations:
            normalized = normalize_annotation_for_comparison(original)
            if normalized not in normalized_annotations:
                normalized_annotations[normalized] = []
            normalized_annotations[normalized].append(original)

        # Count normalized annotations
        normalized_counts = {
            norm: len(originals) for norm, originals in normalized_annotations.items()
        }

        # Find most common normalized annotation
        most_common_normalized = max(normalized_counts.items(), key=lambda x: x[1])
        most_common_norm_key = most_common_normalized[0]
        most_common_count = most_common_normalized[1]

        # Get the most frequent original annotation from the most common normalized group
        original_annotations = normalized_annotations[most_common_norm_key]
        original_counts = Counter(original_annotations)
        most_common_annotation = original_counts.most_common(1)[0][0]

        # Calculate consensus proportion based on normalized counts
        prop = most_common_count / len(cluster_annotations)

        # Calculate entropy based on normalized groups
        ent = 0.0
        total = len(cluster_annotations)
        for count in normalized_counts.values():
            p = count / total
            ent -= p * (math.log2(p) if p > 0 else 0)

        # Check if simple consensus meets thresholds
        if prop >= consensus_threshold and ent <= entropy_threshold:
            # Simple consensus is sufficient
            consensus_proportion[cluster] = prop
            entropy[cluster] = ent
            consensus[cluster] = most_common_annotation
            write_log(
                f"Cluster {cluster} achieved consensus with simple check: "
                f"CP={prop:.2f}, H={ent:.2f}",
                level="info",
            )
            continue

        # Simple consensus didn't meet thresholds, use LLM for double-checking
        write_log(
            f"Cluster {cluster} needs LLM double-check: CP={prop:.2f}, H={ent:.2f}",
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
        primary_api_key = _get_api_key(primary_provider, api_keys)

        # If primary model is not available and we have available_models, try to use one of them
        if not primary_api_key and available_models and not consensus_model:
            write_log(
                f"Primary consensus model {primary_provider} not available, trying available models",
                level="info",
            )

            from .functions import get_provider

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
            fallback_provider="anthropic",
            fallback_model="claude-3-5-sonnet-latest",
            api_keys=api_keys,
        )

        # Parse LLM response using unified parser
        if llm_response:
            prop_value, entropy_value, majority_prediction = _parse_llm_consensus_response(
                llm_response
            )

            if (
                prop_value is not None
                and entropy_value is not None
                and majority_prediction is not None
            ):
                consensus_proportion[cluster] = prop_value
                entropy[cluster] = entropy_value
                consensus[cluster] = majority_prediction
                write_log(
                    f"LLM consensus check for cluster {cluster}: "
                    f"CP={prop_value:.2f}, H={entropy_value:.2f}",
                    level="info",
                )
                continue

        # If LLM failed, use the simple consensus results we already calculated
        consensus_proportion[cluster] = prop
        entropy[cluster] = ent
        consensus[cluster] = most_common_annotation
        write_log(
            f"Using simple consensus for cluster {cluster} after LLM failure: "
            f"CP={prop:.2f}, H={ent:.2f}",
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


def process_controversial_clusters(
    marker_genes: dict[str, list[str]],
    controversial_clusters: list[str],
    model_predictions: dict[str, dict[str, str]],
    species: str,
    tissue: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_discussion_rounds: int = 3,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, float], dict[str, float]]:
    """Process controversial clusters by facilitating a discussion between models.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        controversial_clusters: List of controversial cluster IDs
        model_predictions: Dictionary mapping model names to dictionaries of
            cluster annotations
        species: Species name (e.g., 'human', 'mouse')
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        provider: LLM provider for the discussion
        model: Model name for the discussion
        api_key: API key for the provider
        max_discussion_rounds: Maximum number of discussion rounds for controversial clusters
        consensus_threshold: Agreement threshold for determining when consensus is reached
        entropy_threshold: Entropy threshold for determining when consensus is reached
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files

    Returns:
        tuple[dict[str, str], dict[str, list[str]], dict[str, float], dict[str, float]]:
            - Dictionary mapping cluster IDs to resolved annotations
            - Dictionary mapping cluster IDs to discussion history for each round
            - Dictionary mapping cluster IDs to updated consensus proportion scores
            - Dictionary mapping cluster IDs to updated entropy scores

    """

    from .prompts import create_consensus_check_prompt

    results = {}
    discussion_history = {}
    updated_consensus_proportion = {}
    updated_entropy = {}

    for cluster_id in controversial_clusters:
        write_log(f"Processing controversial cluster {cluster_id}")

        # Get marker genes for this cluster
        cluster_markers = marker_genes.get(cluster_id, [])
        if not cluster_markers:
            write_log(
                f"Warning: No marker genes found for cluster {cluster_id}",
                level="warning",
            )
            results[cluster_id] = "Unknown (no markers)"
            discussion_history[cluster_id] = ["No marker genes found for this cluster"]
            continue

        # Get model predictions for this cluster
        model_votes = {
            model: predictions.get(cluster_id, "Unknown")
            for model, predictions in model_predictions.items()
            if cluster_id in predictions
        }

        # Use a more capable model for discussion if possible
        discussion_model = model
        if provider == "openai" and not discussion_model:
            discussion_model = "gpt-4o"
        elif provider == "anthropic" and not discussion_model:
            discussion_model = "claude-3-opus"

        # Initialize variables for iterative discussion
        current_round = 1
        consensus_reached = False
        final_decision = None
        rounds_history = []
        current_votes = model_votes.copy()

        # Create initial consensus check prompt for LLM to calculate metrics

        # Get all annotations for this cluster
        annotations = list(current_votes.values())

        # Create prompt for LLM to check consensus
        consensus_check_prompt = create_consensus_check_prompt(annotations)

        # Get response from LLM
        consensus_check_response = get_model_response(
            consensus_check_prompt,
            provider,
            discussion_model,
            api_key,
            use_cache,
            cache_dir,
        )

        # Parse response to get consensus metrics
        try:
            lines = consensus_check_response.strip().split("\n")
            if len(lines) >= 3:
                # Extract consensus proportion
                cp = float(lines[1].strip())

                # Extract entropy value
                h = float(lines[2].strip())

                write_log(
                    f"Initial metrics for cluster {cluster_id} (LLM calculated): CP={cp:.2f}, H={h:.2f}"
                )
            else:
                # Fallback if LLM response format is unexpected
                cp = 0.25  # Low consensus to ensure discussion happens
                h = 2.0  # High entropy to indicate uncertainty
                write_log(
                    f"Could not parse LLM consensus check response, using default values: CP={cp:.2f}, H={h:.2f}",
                    level="warning",
                )
        except (ValueError, IndexError, AttributeError, TypeError) as e:
            # Fallback if parsing fails
            cp = 0.25  # Low consensus to ensure discussion happens
            h = 2.0  # High entropy to indicate uncertainty
            write_log(
                f"Error parsing LLM consensus check response: {str(e)}, using default values: CP={cp:.2f}, H={h:.2f}",
                level="warning",
            )

        rounds_history.append(
            f"Initial votes: {current_votes}\nConsensus Proportion (CP): {cp:.2f}\nShannon Entropy (H): {h:.2f}"
        )

        # Start iterative discussion process
        try:
            while current_round <= max_discussion_rounds and not consensus_reached:
                write_log(f"Starting discussion round {current_round} for cluster {cluster_id}")

                # Generate discussion prompt based on current round
                if current_round == 1:
                    # Initial discussion round
                    prompt = create_discussion_prompt(
                        cluster_id=cluster_id,
                        marker_genes=cluster_markers,
                        model_votes=current_votes,
                        species=species,
                        tissue=tissue,
                    )
                else:
                    # Follow-up rounds include previous discussion
                    prompt = create_discussion_prompt(
                        cluster_id=cluster_id,
                        marker_genes=cluster_markers,
                        model_votes=current_votes,
                        species=species,
                        tissue=tissue,
                        previous_discussion=rounds_history[-1],
                    )

                # Get response for this round
                response = get_model_response(
                    prompt, provider, discussion_model, api_key, use_cache, cache_dir
                )

                # Extract potential decision from this round
                round_decision = extract_cell_type_from_discussion(response)

                # Record this round's discussion
                round_summary = f"Round {current_round} Discussion:\n{response}\n\nProposed cell type: {round_decision or 'Unclear'}"
                rounds_history.append(round_summary)

                # Check if we've reached consensus
                if current_round < max_discussion_rounds and round_decision:
                    # Create a consensus check prompt
                    consensus_prompt = create_discussion_consensus_check_prompt(
                        cluster_id=cluster_id,
                        discussion=response,
                        proposed_cell_type=round_decision,
                    )

                    # Get consensus check response
                    consensus_response = get_model_response(
                        consensus_prompt,
                        provider,
                        discussion_model,
                        api_key,
                        use_cache,
                        cache_dir,
                    )

                    # Add consensus checker result to history
                    rounds_history.append(f"Consensus Check {current_round}:\n{consensus_response}")

                    # Previously had consensus indicators check here, now using metrics extraction

                    # Extract consensus proportion and entropy values for the current round
                    cp_value, h_value = extract_consensus_metrics_from_discussion(response)

                    # If unable to extract from discussion, try to extract from consensus check response
                    if cp_value is None or h_value is None:
                        cp_value, h_value = extract_consensus_metrics_from_discussion(
                            consensus_response
                        )

                    # If still unable to extract, use default values
                    if cp_value is None:
                        cp_value = 0.5  # Default medium consensus proportion
                        write_log(
                            f"Could not extract consensus proportion for cluster {cluster_id} "
                            f"in round {current_round}, using default value: {cp_value}",
                            level="warning",
                        )

                    if h_value is None:
                        h_value = 1.0  # Default medium entropy value
                        write_log(
                            f"Could not extract entropy for cluster {cluster_id} "
                            f"in round {current_round}, using default value: {h_value}",
                            level="warning",
                        )

                    # Use consensus proportion and entropy values to compare with thresholds
                    consensus_reached = (
                        cp_value >= consensus_threshold and h_value <= entropy_threshold
                    )
                    write_log(
                        f"Consensus check for cluster {cluster_id} in round {current_round}: "
                        f"CP={cp_value:.2f}, H={h_value:.2f}, threshold CP>={consensus_threshold:.2f}, "
                        f"H<={entropy_threshold:.2f}",
                        level="info",
                    )

                    if consensus_reached:
                        final_decision = round_decision
                        write_log(
                            f"Consensus reached for cluster {cluster_id} in round {current_round}",
                            level="info",
                        )

                        # Extract CP and H from the discussion if available
                        cp_value, h_value = extract_consensus_metrics_from_discussion(response)
                        if cp_value is not None and h_value is not None:
                            updated_consensus_proportion[cluster_id] = cp_value
                            updated_entropy[cluster_id] = h_value
                        else:
                            # If not found in discussion, set high consensus values
                            updated_consensus_proportion[cluster_id] = 1.0
                            updated_entropy[cluster_id] = 0.0

                        rounds_history.append(
                            f"Consensus reached in round {current_round}\n"
                            f"Final cell type: {final_decision}\n"
                            f"Consensus Proportion (CP): {updated_consensus_proportion[cluster_id]:.2f}\n"
                            f"Shannon Entropy (H): {updated_entropy[cluster_id]:.2f}"
                        )

                # Move to next round if no consensus yet
                if not consensus_reached:
                    current_round += 1

            # After all rounds, use the last round's decision if no consensus was reached
            if not final_decision:
                # Try to extract majority_prediction from the last consensus check
                # Only try to access consensus_response if it was actually created in this iteration
                last_consensus_check = None
                if rounds_history and len(rounds_history) >= 1 and "consensus_response" in locals():
                    # Get the response from the last consensus check
                    last_consensus_check = consensus_response

                if last_consensus_check is not None:
                    # Try to extract majority_prediction
                    try:
                        lines = last_consensus_check.strip().split("\n")
                        lines = [line.strip() for line in lines if line.strip()]

                        # If it's the standard format (4 lines), the 4th line should be the
                        # majority_prediction
                        if (
                            len(lines) >= 4
                            and re.match(r"^\s*[01]\s*$", lines[0])
                            and re.match(r"^\s*(0\.\d+|1\.0*|1)\s*$", lines[1])
                        ):
                            majority_prediction = lines[3].strip()
                            if majority_prediction and majority_prediction != "Unknown":
                                final_decision = clean_annotation(majority_prediction)
                                write_log(
                                    f"Using majority prediction from last consensus check "
                                    f"for cluster {cluster_id}: {final_decision}",
                                    level="info",
                                )
                    except (KeyError, ValueError, AttributeError, IndexError) as e:
                        write_log(
                            f"Error extracting majority prediction: {str(e)}",
                            level="warning",
                        )

                # If unable to extract majority_prediction, use the decision from the
                # last round
                if not final_decision and round_decision:
                    final_decision = round_decision
                    write_log(
                        f"Using final round decision for cluster {cluster_id} "
                        f"after {max_discussion_rounds} rounds",
                        level="info",
                    )

            # Store the final result
            if not final_decision:
                write_log(
                    f"Warning: Could not reach a decision for cluster {cluster_id} "
                    f"after {max_discussion_rounds} rounds",
                    level="warning",
                )
                results[cluster_id] = "Inconclusive"
                # For inconclusive results, extract metrics from the last round
                # if available
                if rounds_history:
                    last_round = rounds_history[-1]
                    cp_value, h_value = extract_consensus_metrics_from_discussion(last_round)
                    if cp_value is not None and h_value is not None:
                        updated_consensus_proportion[cluster_id] = cp_value
                        updated_entropy[cluster_id] = h_value
                    else:
                        # If not found, set high uncertainty values
                        updated_consensus_proportion[cluster_id] = 0.5
                        updated_entropy[cluster_id] = 1.0
                else:
                    # If no discussion history, set high uncertainty values
                    updated_consensus_proportion[cluster_id] = 0.5
                    updated_entropy[cluster_id] = 1.0
            else:
                results[cluster_id] = final_decision
                # If consensus wasn't explicitly reached but we have a final decision
                # Extract metrics from the last round if available
                if cluster_id not in updated_consensus_proportion and rounds_history:
                    last_round = rounds_history[-1]
                    cp_value, h_value = extract_consensus_metrics_from_discussion(last_round)
                    if cp_value is not None and h_value is not None:
                        updated_consensus_proportion[cluster_id] = cp_value
                        updated_entropy[cluster_id] = h_value
                    else:
                        # If not found, set reasonable default values
                        updated_consensus_proportion[cluster_id] = 0.75
                        updated_entropy[cluster_id] = 0.5

            # Store the full discussion history
            discussion_history[cluster_id] = rounds_history

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            AttributeError,
        ) as e:
            write_log(
                f"Error during discussion for cluster {cluster_id}: {str(e)}",
                level="error",
            )
            results[cluster_id] = f"Error during discussion: {str(e)}"
            discussion_history[cluster_id] = [f"Error occurred: {str(e)}"]

    return results, discussion_history, updated_consensus_proportion, updated_entropy


def extract_consensus_metrics_from_discussion(
    discussion: str,
) -> tuple[Optional[float], Optional[float]]:
    """Extract consensus proportion (CP) and entropy (H) values from discussion text.

    Args:
        discussion: Text of the model discussion

    Returns:
        tuple[Optional[float], Optional[float]]: Extracted CP and H values, or None if not found

    """
    # First try to extract from structured format (4 lines)
    lines = discussion.strip().split("\n")
    # Clean up lines
    lines = [line.strip() for line in lines if line.strip()]

    # If we have at least 3 lines, try to extract from structured format
    if len(lines) >= 3:
        try:
            # Line 2 should be CP value
            cp_value = float(lines[1])
            # Line 3 should be H value
            h_value = float(lines[2])
            return cp_value, h_value
        except (ValueError, IndexError):
            # If structured format fails, continue with regex
            pass

    # Fallback to regex patterns
    cp_pattern = r"(?i)consensus\s+proportion\s*(?:\(CP\))?\s*[:=]\s*([0-9.]+)"
    h_pattern = r"(?i)(?:shannon\s+)?entropy\s*(?:\(H\))?\s*[:=]\s*([0-9.]+)"

    cp_value = None
    h_value = None

    # Find CP value
    cp_match = re.search(cp_pattern, discussion)
    if cp_match:
        with contextlib.suppress(ValueError, IndexError):
            cp_value = float(cp_match.group(1))

    # Find H value
    h_match = re.search(h_pattern, discussion)
    if h_match:
        with contextlib.suppress(ValueError, IndexError):
            h_value = float(h_match.group(1))

    return cp_value, h_value


def extract_cell_type_from_discussion(discussion: str) -> Optional[str]:
    """Extract the final cell type determination from a discussion.

    Args:
        discussion: Text of the model discussion

    Returns:
        Optional[str]: Extracted cell type or None if not found

    """
    # Look for common patterns in discussion summaries
    patterns = [
        r"(?i)final\s+cell\s+type\s+determination:?\s*(.*)",
        r"(?i)final\s+decision:?\s*(.*)",
        r"(?i)conclusion:?\s*(.*)",
        r"(?i)the\s+best\s+annotation\s+is:?\s*(.*)",
        r"(?i)I\s+conclude\s+that\s+this\s+cluster\s+(?:is|represents)\s+(.*)",
        r"(?i)based\s+on\s+[^,]+,\s+this\s+cluster\s+is\s+(.*)",
        r"(?i)proposed\s+cell\s+type:?\s*(.*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, discussion)
        if match:
            # Clean up the result
            result = match.group(1).strip()

            # Remove trailing punctuation
            if result and result[-1] in [".", ",", ";"]:
                result = result[:-1].strip()

            # Remove quotes if present
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1].strip()

            # Skip invalid results
            if result.lower() in ["unclear", "none", "n/a", "on cell type"]:
                continue

            return result

    # If no match with specific patterns, look for the last line that mentions "cell" or "type"
    lines = discussion.strip().split("\n")
    for line in reversed(lines):
        if "cell" in line.lower() or "type" in line.lower():
            # Try to extract a short phrase
            if ":" in line:
                parts = line.split(":", 1)
                result = parts[1].strip()
                # Skip invalid results
                if result.lower() in ["unclear", "none", "n/a", "on cell type"]:
                    continue
                return result
            result = line.strip()
            # Skip invalid results
            if result.lower() in ["unclear", "none", "n/a", "on cell type"]:
                continue
            return result

    return None


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

    Returns:
        dict[str, Any]: Dictionary containing consensus results and metadata

    """
    from .annotate import annotate_clusters
    from .functions import get_provider

    # Set up logging
    if verbose:
        write_log("Starting interactive consensus annotation")

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
                from .utils import load_api_key

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
            from .utils import load_api_key

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
        # The model name is already in the correct format (e.g., "openai/gpt-4o")
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
                use_cache=use_cache,
                cache_dir=cache_dir,
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
            write_log(f"Error annotating with {model_name}: {str(e)}", level="error")

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

    # If there are controversial clusters, resolve them
    resolved = {}
    if controversial:
        # Choose model for discussion
        discussion_model = None
        discussion_provider = None

        # First priority: use consensus_model if specified
        if consensus_model_dict:
            discussion_provider = consensus_model_dict.get("provider")
            discussion_model = consensus_model_dict.get("model")
            if verbose:
                write_log(f"Using specified consensus model for discussion: {discussion_model}")
        else:
            # Otherwise, try to use the most capable model available from the input models
            for preferred_model_name in ["gpt-4o", "claude-3-opus", "gemini-2.0-pro"]:
                # Check if the preferred model is in the models list
                for model_item in models:
                    if isinstance(model_item, dict):
                        # For dictionary models, check the 'model' key
                        if model_item.get("model") == preferred_model_name:
                            discussion_provider = model_item.get("provider")
                            discussion_model = preferred_model_name
                            # If provider is not explicitly provided, try to get it from model name
                            if not discussion_provider:
                                discussion_provider = get_provider(discussion_model)
                            if discussion_provider in api_keys:
                                break
                    elif model_item == preferred_model_name:
                        # For string models
                        provider = get_provider(preferred_model_name)
                        if provider in api_keys:
                            discussion_model = preferred_model_name
                            discussion_provider = provider
                            break
                # If we found a model, break out of the outer loop too
                if discussion_model:
                    break

            # If no preferred model is available, use the first one
            if not discussion_model and models:
                first_model = models[0]
                # Handle both string models and dict models
                if isinstance(first_model, dict):
                    discussion_provider = first_model.get("provider")
                    discussion_model = first_model.get("model")

                    # If provider is not explicitly provided, try to get it from model name
                    if not discussion_provider and discussion_model:
                        discussion_provider = get_provider(discussion_model)
                else:
                    discussion_model = first_model
                    discussion_provider = get_provider(discussion_model)

        if discussion_model:
            if verbose:
                write_log(f"Resolving controversial clusters using {discussion_model}")

            try:
                resolved, discussion_logs, updated_cp, updated_h = process_controversial_clusters(
                    marker_genes=marker_genes,
                    controversial_clusters=controversial,
                    model_predictions=model_results,
                    species=species,
                    tissue=tissue,
                    provider=discussion_provider,
                    model=discussion_model,
                    api_key=api_keys.get(discussion_provider),
                    max_discussion_rounds=max_discussion_rounds,
                    consensus_threshold=consensus_threshold,
                    entropy_threshold=entropy_threshold,
                    use_cache=use_cache,
                    cache_dir=cache_dir,
                )

                # Update consensus proportion and entropy for resolved clusters
                for cluster_id, cp in updated_cp.items():
                    consensus_proportion[cluster_id] = cp

                for cluster_id, h in updated_h.items():
                    entropy[cluster_id] = h

                if verbose:
                    write_log(f"Successfully resolved {len(resolved)} controversial clusters")
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                json.JSONDecodeError,
                AttributeError,
            ) as e:
                write_log(f"Error resolving controversial clusters: {str(e)}", level="error")

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
        "discussion_logs": discussion_logs if "discussion_logs" in locals() else {},
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


def print_consensus_summary(result: dict[str, Any]) -> None:
    """Print a summary of consensus annotation results.

    Args:
        result: Dictionary containing consensus results from interactive_consensus_annotation

    """
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\n=== CONSENSUS ANNOTATION SUMMARY ===\n")

    # Print metadata
    metadata = result.get("metadata", {})
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"Species: {metadata.get('species', 'Unknown')}")
    if metadata.get("tissue"):
        print(f"Tissue: {metadata['tissue']}")
    models_list = metadata.get("models", [])
    model_names = []
    for model in models_list:
        if isinstance(model, dict):
            model_names.append(model.get("model", "Unknown"))
        else:
            model_names.append(str(model))
    print(f"Models used: {', '.join(model_names)}")
    print(f"Consensus threshold: {metadata.get('consensus_threshold', 0.6)}")
    print()

    # Print controversial clusters
    controversial = result.get("controversial_clusters", [])
    if controversial:
        print(f"Controversial clusters: {len(controversial)} - {', '.join(controversial)}")
    else:
        print("No controversial clusters found.")
    print()

    # Print consensus annotations with consensus proportion and entropy
    consensus = result.get("consensus", {})
    consensus_proportion = result.get("consensus_proportion", {})
    entropy = result.get("entropy", {})
    resolved = result.get("resolved", {})

    print("Cluster annotations:")
    for cluster, annotation in sorted(consensus.items(), key=lambda x: x[0]):
        cp = consensus_proportion.get(cluster, 0)
        ent = entropy.get(cluster, 0)
        if cluster in resolved:
            # For resolved clusters, show CP and H if available in the discussion logs
            discussion_logs = result.get("discussion_logs", {})
            cp_value = "N/A"
            h_value = "N/A"

            # Try to extract CP and H from discussion logs
            if cluster in discussion_logs:
                logs = discussion_logs[cluster]
                # Check if logs is a list or string
                # Convert logs to string if it's a list, otherwise use directly
                logs_text = "\n".join(logs) if isinstance(logs, list) else logs

                # Look for CP and H in the last round
                for line in reversed(logs_text.split("\n")):
                    if "Consensus Proportion (CP):" in line:
                        cp_parts = line.split("Consensus Proportion (CP):")[1].strip().split()
                        if cp_parts:
                            cp_value = cp_parts[0]
                    if "Shannon Entropy (H):" in line:
                        h_parts = line.split("Shannon Entropy (H):")[1].strip().split()
                        if h_parts:
                            h_value = h_parts[0]

            print(f"  Cluster {cluster}: {annotation} [Resolved, CP: {cp_value}, H: {h_value}]")
        else:
            # For non-resolved clusters, use the calculated CP and entropy values
            cp_value = cp
            h_value = ent

            # Display different messages based on agreement level
            # Use the already calculated entropy value
            print(f"  Cluster {cluster}: {annotation} [CP: {cp_value:.2f}, H: {h_value:.2f}]")
    print()

    # Print model annotations comparison for controversial clusters
    if controversial:
        print("\nModel annotations for controversial clusters:")
        model_annotations = result.get("model_annotations", {})
        models = list(model_annotations.keys())

        for cluster in controversial:
            print(f"\nCluster {cluster}:")
            for model in models:
                if cluster in model_annotations.get(model, {}):
                    print(f"  {model}: {model_annotations[model].get(cluster, 'Unknown')}")
            if cluster in resolved:
                print(f"  RESOLVED: {resolved[cluster]}")
            print()


def facilitate_cluster_discussion(
    cluster_id: str,
    marker_genes: list[str],
    model_votes: dict[str, str],
    species: str,
    tissue: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> str:
    """Facilitate a discussion between different model predictions for a controversial cluster.

    Args:
        cluster_id: ID of the cluster
        marker_genes: List of marker genes for the cluster
        model_votes: Dictionary mapping model names to cell type annotations
        species: Species name (e.g., 'human', 'mouse')
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        provider: LLM provider for the discussion
        model: Model name for the discussion
        api_key: API key for the provider
        use_cache: Whether to use cache

    Returns:
        str: Discussion result

    """
    from .prompts import create_discussion_prompt

    # Generate discussion prompt
    prompt = create_discussion_prompt(
        cluster_id=cluster_id,
        marker_genes=marker_genes,
        model_votes=model_votes,
        species=species,
        tissue=tissue,
    )

    # Get response
    response = get_model_response(prompt, provider, model, api_key, use_cache)

    # Extract final decision
    cell_type = extract_cell_type_from_discussion(response)

    # Return the full discussion and the extracted cell type
    return f"{response}\n\nFINAL DETERMINATION: {cell_type}"


def summarize_discussion(discussion: str) -> str:
    """Summarize a model discussion about cell type annotation.

    Args:
        discussion: Full discussion text

    Returns:
        str: Summary of the discussion

    """
    # Extract key points from the discussion
    lines = discussion.strip().split("\n")
    summary_lines = []

    # Look for common summary indicators
    for line in lines:
        line = line.strip()
        if line.lower().startswith(
            ("conclusion", "summary", "final", "therefore", "overall", "in summary")
        ):
            summary_lines.append(line)

    # If we found summary lines, join them
    if summary_lines:
        return "\n".join(summary_lines)

    # Otherwise, extract the final decision
    cell_type = extract_cell_type_from_discussion(discussion)
    if cell_type:
        return f"Final cell type determination: {cell_type}"

    # If all else fails, return the last few lines
    return "\n".join(lines[-3:])
