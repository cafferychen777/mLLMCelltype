"""
Module for consensus annotation of cell types from multiple LLM predictions.
"""

import os
import time
import json
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import re
from collections import Counter

from .logger import write_log
from .utils import clean_annotation, find_agreement
from .functions import identify_controversial_clusters
from .prompts import create_discussion_prompt, create_discussion_consensus_check_prompt

def check_consensus_with_llm(predictions: Dict[str, Dict[str, str]], api_keys: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]:
    """
    Check consensus among different model predictions using LLM assistance.
    This function uses an LLM (Qwen or Claude) to evaluate semantic similarity between annotations.
    
    Args:
        predictions: Dictionary mapping model names to dictionaries of cluster annotations
        api_keys: Dictionary mapping provider names to API keys
        
    Returns:
        Tuple of:
            - Dictionary mapping cluster IDs to consensus annotations
            - Dictionary mapping cluster IDs to consensus proportion scores
            - Dictionary mapping cluster IDs to entropy scores
    """
    from .annotate import get_model_response
    from .prompts import create_consensus_check_prompt
    from .functions import get_provider
    import math
    from collections import Counter
    
    consensus = {}
    consensus_proportion = {}
    entropy = {}
    
    # Ensure we have annotations
    if not predictions or not all(predictions.values()):
        return {}, {}, {}
    
    # Get all clusters
    all_clusters = set()
    for model_results in predictions.values():
        all_clusters.update(model_results.keys())
    
    # Process each cluster
    for cluster in all_clusters:
        # Collect all annotations for this cluster
        cluster_annotations = []
        
        for model, results in predictions.items():
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
        
        # Create prompt for LLM
        prompt = create_consensus_check_prompt(cluster_annotations)
        
        # Try with Qwen first
        max_retries = 3
        llm_response = None
        
        # First try with Qwen
        for attempt in range(max_retries):
            try:
                # Get API key
                qwen_api_key = None
                if api_keys and 'qwen' in api_keys:
                    qwen_api_key = api_keys['qwen']
                
                if not qwen_api_key:
                    from .utils import load_api_key
                    qwen_api_key = load_api_key('qwen')
                
                if qwen_api_key:
                    llm_response = get_model_response(
                        prompt=prompt,
                        provider='qwen',
                        model='qwen-max-2025-01-25',
                        api_key=qwen_api_key
                    )
                    write_log(f"Successfully got response from Qwen on attempt {attempt+1}")
                    break
                else:
                    write_log("No Qwen API key found, trying Claude")
                    break
            except Exception as e:
                write_log(f"Error on Qwen attempt {attempt+1}: {str(e)}", level='warning')
                if attempt == max_retries - 1:
                    write_log("All Qwen retry attempts failed, falling back to Claude")
                else:
                    write_log(f"Waiting before next attempt...")
                    time.sleep(5 * (2 ** attempt))
        
        # Try Claude as fallback
        if not llm_response:
            try:
                # Get API key
                anthropic_api_key = None
                if api_keys and 'anthropic' in api_keys:
                    anthropic_api_key = api_keys['anthropic']
                
                if not anthropic_api_key:
                    from .utils import load_api_key
                    anthropic_api_key = load_api_key('anthropic')
                
                if anthropic_api_key:
                    llm_response = get_model_response(
                        prompt=prompt,
                        provider='anthropic',
                        model='claude-3-5-sonnet-latest',
                        api_key=anthropic_api_key
                    )
                    write_log("Successfully got response from Claude as fallback")
                else:
                    write_log("No Claude API key found, falling back to simple consensus")
            except Exception as e:
                write_log(f"Error on Claude fallback: {str(e)}", level='warning')
        
        # Parse LLM response
        if llm_response:
            try:
                # Split response by newlines and clean up
                lines = llm_response.strip().split('\n')
                lines = [line.strip() for line in lines if line.strip()]
                
                # Get the last 4 non-empty lines (standard format)
                if len(lines) >= 4:
                    result_lines = lines[-4:]
                    
                    # Check if it's a standard format (0/1, proportion, entropy, annotation)
                    if (re.match(r'^\s*[01]\s*$', result_lines[0]) and 
                        re.match(r'^\s*(0\.\d+|1\.0*|1)\s*$', result_lines[1]) and 
                        re.match(r'^\s*(\d+\.\d+|\d+)\s*$', result_lines[2])):
                        
                        # Extract consensus indicator
                        consensus_reached = int(result_lines[0].strip()) == 1
                        
                        # Extract consensus proportion
                        prop_value = float(result_lines[1].strip())
                        consensus_proportion[cluster] = prop_value
                        
                        # Extract entropy value
                        entropy_value = float(result_lines[2].strip())
                        entropy[cluster] = entropy_value
                        
                        # Extract majority prediction
                        majority_prediction = result_lines[3].strip()
                        consensus[cluster] = majority_prediction if majority_prediction and majority_prediction != "Unknown" else "Unknown"
                        
                        continue  # Successfully parsed, move to next cluster
            except Exception as e:
                write_log(f"Error parsing LLM response: {str(e)}", level='warning')
        
        # Fallback to simple consensus calculation if LLM approach failed
        # Count occurrences of each annotation
        annotation_counts = Counter(cluster_annotations)
        
        # Find most common annotation
        most_common = annotation_counts.most_common(1)[0]
        most_common_annotation = most_common[0]
        most_common_count = most_common[1]
        
        # Calculate consensus proportion
        prop = most_common_count / len(cluster_annotations)
        consensus_proportion[cluster] = prop
        
        # Calculate entropy
        ent = 0.0
        total = len(cluster_annotations)
        for count in annotation_counts.values():
            p = count / total
            ent -= p * (math.log2(p) if p > 0 else 0)
        entropy[cluster] = ent
        
        consensus[cluster] = most_common_annotation
    
    return consensus, consensus_proportion, entropy

def check_consensus(predictions: Dict[str, Dict[str, str]], consensus_threshold: float = 0.6, entropy_threshold: float = 1.0, api_keys: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float], List[str]]:
    """
    Check if there is consensus among different model predictions.
    Uses LLM assistance to evaluate semantic similarity between annotations.
    
    Args:
        predictions: Dictionary mapping model names to dictionaries of cluster annotations
        consensus_threshold: Agreement threshold below which a cluster is considered controversial
        entropy_threshold: Entropy threshold above which a cluster is considered controversial
        api_keys: Dictionary mapping provider names to API keys
        
    Returns:
        Tuple of:
            - Dictionary mapping cluster IDs to consensus annotations
            - Dictionary mapping cluster IDs to consensus proportion scores
            - Dictionary mapping cluster IDs to entropy scores
            - List of controversial cluster IDs
    """
    # Find consensus annotations and metrics using LLM
    consensus, consensus_proportion, entropy = check_consensus_with_llm(predictions, api_keys)
    
    # Find controversial clusters based on both consensus proportion and entropy
    controversial = [cluster for cluster, score in consensus_proportion.items() 
                   if score < consensus_threshold or entropy.get(cluster, 0) > entropy_threshold]
    
    return consensus, consensus_proportion, entropy, controversial

def process_controversial_clusters(
    marker_genes: Dict[str, List[str]],
    controversial_clusters: List[str],
    model_predictions: Dict[str, Dict[str, str]],
    species: str,
    tissue: Optional[str] = None,
    provider: str = 'openai',
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_discussion_rounds: int = 3,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    use_cache: bool = True,
    cache_dir: Optional[str] = None
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, float], Dict[str, float]]:
    """
    Process controversial clusters by facilitating a discussion between models.
    
    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        controversial_clusters: List of controversial cluster IDs
        model_predictions: Dictionary mapping model names to dictionaries of cluster annotations
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
        Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, float], Dict[str, float]]: 
            - Dictionary mapping cluster IDs to resolved annotations
            - Dictionary mapping cluster IDs to discussion history for each round
            - Dictionary mapping cluster IDs to updated consensus proportion scores
            - Dictionary mapping cluster IDs to updated entropy scores
    """
    from .annotate import get_model_response
    from .prompts import create_consensus_check_prompt
    import math
    from collections import Counter
    
    results = {}
    discussion_history = {}
    updated_consensus_proportion = {}
    updated_entropy = {}
    
    for cluster_id in controversial_clusters:
        write_log(f"Processing controversial cluster {cluster_id}")
        
        # Get marker genes for this cluster
        cluster_markers = marker_genes.get(cluster_id, [])
        if not cluster_markers:
            write_log(f"Warning: No marker genes found for cluster {cluster_id}", level='warning')
            results[cluster_id] = "Unknown (no markers)"
            discussion_history[cluster_id] = ["No marker genes found for this cluster"]
            continue
        
        # Get model predictions for this cluster
        model_votes = {model: predictions.get(cluster_id, "Unknown") 
                   for model, predictions in model_predictions.items() 
                   if cluster_id in predictions}
        
        # Use a more capable model for discussion if possible
        discussion_model = model
        if provider == 'openai' and not discussion_model:
            discussion_model = 'gpt-4o'
        elif provider == 'anthropic' and not discussion_model:
            discussion_model = 'claude-3-opus'
        
        # Initialize variables for iterative discussion
        current_round = 1
        consensus_reached = False
        final_decision = None
        rounds_history = []
        current_votes = model_votes.copy()
    
        # Create initial consensus check prompt for LLM to calculate metrics
        from .prompts import create_consensus_check_prompt
        
        # Get all annotations for this cluster
        annotations = list(current_votes.values())
        
        # Create prompt for LLM to check consensus
        consensus_check_prompt = create_consensus_check_prompt(annotations)
        
        # Get response from LLM
        consensus_check_response = get_model_response(consensus_check_prompt, provider, discussion_model, api_key, use_cache, cache_dir)
        
        # Parse response to get consensus metrics
        try:
            lines = consensus_check_response.strip().split('\n')
            if len(lines) >= 3:
                # Extract consensus indicator (0 or 1)
                consensus_reached_initial = int(lines[0].strip()) == 1
                
                # Extract consensus proportion
                cp = float(lines[1].strip())
                
                # Extract entropy value
                h = float(lines[2].strip())
                
                write_log(f"Initial metrics for cluster {cluster_id} (LLM calculated): CP={cp:.2f}, H={h:.2f}")
            else:
                # Fallback if LLM response format is unexpected
                cp = 0.25  # Low consensus to ensure discussion happens
                h = 2.0    # High entropy to indicate uncertainty
                write_log(f"Could not parse LLM consensus check response, using default values: CP={cp:.2f}, H={h:.2f}", level='warning')
        except Exception as e:
            # Fallback if parsing fails
            cp = 0.25  # Low consensus to ensure discussion happens
            h = 2.0    # High entropy to indicate uncertainty
            write_log(f"Error parsing LLM consensus check response: {str(e)}, using default values: CP={cp:.2f}, H={h:.2f}", level='warning')
        
        rounds_history.append(f"Initial votes: {current_votes}\nConsensus Proportion (CP): {cp:.2f}\nShannon Entropy (H): {h:.2f}")
        
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
                        tissue=tissue
                    )
                else:
                    # Follow-up rounds include previous discussion
                    prompt = create_discussion_prompt(
                        cluster_id=cluster_id,
                        marker_genes=cluster_markers,
                        model_votes=current_votes,
                        species=species,
                        tissue=tissue,
                        previous_discussion=rounds_history[-1]
                    )
                
                # Get response for this round
                response = get_model_response(prompt, provider, discussion_model, api_key, use_cache, cache_dir)
            
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
                        proposed_cell_type=round_decision
                    )
                    
                    # Get consensus check response
                    consensus_response = get_model_response(consensus_prompt, provider, discussion_model, api_key, use_cache, cache_dir)
                    
                    # Add consensus checker result to history
                    rounds_history.append(f"Consensus Check {current_round}:\n{consensus_response}")
                    
                    # Check if consensus is reached - using stricter criteria
                    consensus_indicators = [
                        "consensus reached", 
                        "confident in the annotation",
                        "sufficient evidence",
                        "clear determination",
                        "agreement on"
                    ]
                    
                    # Extract consensus proportion and entropy values for the current round
                    cp_value, h_value = extract_consensus_metrics_from_discussion(response)
                    
                    # If unable to extract from discussion, try to extract from consensus check response
                    if cp_value is None or h_value is None:
                        cp_value, h_value = extract_consensus_metrics_from_discussion(consensus_response)
                    
                    # If still unable to extract, use default values
                    if cp_value is None:
                        cp_value = 0.5  # Default medium consensus proportion
                        write_log(f"Could not extract consensus proportion for cluster {cluster_id} in round {current_round}, using default value: {cp_value}", level='warning')
                    
                    if h_value is None:
                        h_value = 1.0  # Default medium entropy value
                        write_log(f"Could not extract entropy for cluster {cluster_id} in round {current_round}, using default value: {h_value}", level='warning')
                    
                    # Use consensus proportion and entropy values to compare with thresholds
                    consensus_reached = cp_value >= consensus_threshold and h_value <= entropy_threshold
                    write_log(f"Consensus check for cluster {cluster_id} in round {current_round}: CP={cp_value:.2f}, H={h_value:.2f}, threshold CP>={consensus_threshold:.2f}, H<={entropy_threshold:.2f}", level='info')
                    
                    if consensus_reached:
                        final_decision = round_decision
                        write_log(f"Consensus reached for cluster {cluster_id} in round {current_round}", level='info')
                        
                        # Extract CP and H from the discussion if available
                        cp_value, h_value = extract_consensus_metrics_from_discussion(response)
                        if cp_value is not None and h_value is not None:
                            updated_consensus_proportion[cluster_id] = cp_value
                            updated_entropy[cluster_id] = h_value
                        else:
                            # If not found in discussion, set high consensus values
                            updated_consensus_proportion[cluster_id] = 1.0
                            updated_entropy[cluster_id] = 0.0
                            
                        rounds_history.append(f"Consensus reached in round {current_round}\nFinal cell type: {final_decision}\nConsensus Proportion (CP): {updated_consensus_proportion[cluster_id]:.2f}\nShannon Entropy (H): {updated_entropy[cluster_id]:.2f}")
            
                # Move to next round if no consensus yet
                if not consensus_reached:
                    current_round += 1
            
            # After all rounds, use the last round's decision if no consensus was explicitly reached
            if not final_decision:
                # Try to extract majority_prediction from the last consensus check
                if rounds_history and len(rounds_history) >= 1:
                    # Get the response from the last consensus check
                    last_consensus_check = consensus_response
                    
                    # Try to extract majority_prediction
                    try:
                        lines = last_consensus_check.strip().split('\n')
                        lines = [line.strip() for line in lines if line.strip()]
                        
                        # If it's the standard format (4 lines), the 4th line should be the majority_prediction
                        if len(lines) >= 4 and re.match(r'^\s*[01]\s*$', lines[0]) and re.match(r'^\s*(0\.\d+|1\.0*|1)\s*$', lines[1]):
                            majority_prediction = lines[3].strip()
                            if majority_prediction and majority_prediction != "Unknown":
                                final_decision = clean_annotation(majority_prediction)
                                write_log(f"Using majority prediction from last consensus check for cluster {cluster_id}: {final_decision}", level='info')
                    except Exception as e:
                        write_log(f"Error extracting majority prediction: {str(e)}", level='warning')
                
                # If unable to extract majority_prediction, use the decision from the last round
                if not final_decision and round_decision:
                    final_decision = round_decision
                    write_log(f"Using final round decision for cluster {cluster_id} after {max_discussion_rounds} rounds", level='info')
            
            # Store the final result
            if not final_decision:
                write_log(f"Warning: Could not reach a decision for cluster {cluster_id} after {max_discussion_rounds} rounds", level='warning')
                results[cluster_id] = "Inconclusive"
                # For inconclusive results, extract metrics from the last round if available
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
                    
        except Exception as e:
            write_log(f"Error during discussion for cluster {cluster_id}: {str(e)}", level='error')
            results[cluster_id] = f"Error during discussion: {str(e)}"
            discussion_history[cluster_id] = [f"Error occurred: {str(e)}"]
    
    return results, discussion_history, updated_consensus_proportion, updated_entropy

def extract_consensus_metrics_from_discussion(discussion: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract consensus proportion (CP) and entropy (H) values from discussion text.
    
    Args:
        discussion: Text of the model discussion
        
    Returns:
        Tuple[Optional[float], Optional[float]]: Extracted CP and H values, or None if not found
    """
    # First try to extract from structured format (4 lines)
    lines = discussion.strip().split('\n')
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
        try:
            cp_value = float(cp_match.group(1))
        except (ValueError, IndexError):
            pass
    
    # Find H value
    h_match = re.search(h_pattern, discussion)
    if h_match:
        try:
            h_value = float(h_match.group(1))
        except (ValueError, IndexError):
            pass
    
    return cp_value, h_value

def extract_cell_type_from_discussion(discussion: str) -> Optional[str]:
    """
    Extract the final cell type determination from a discussion.
    
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
        r"(?i)proposed\s+cell\s+type:?\s*(.*)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, discussion)
        if match:
            # Clean up the result
            result = match.group(1).strip()
            
            # Remove trailing punctuation
            if result and result[-1] in ['.', ',', ';']:
                result = result[:-1].strip()
                
            # Remove quotes if present
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1].strip()
            
            # Skip invalid results
            if result.lower() in ['unclear', 'none', 'n/a', 'on cell type']:
                continue
                
            return result
    
    # If no match with specific patterns, look for the last line that mentions "cell" or "type"
    lines = discussion.strip().split('\n')
    for line in reversed(lines):
        if "cell" in line.lower() or "type" in line.lower():
            # Try to extract a short phrase
            if ":" in line:
                parts = line.split(":", 1)
                result = parts[1].strip()
                # Skip invalid results
                if result.lower() in ['unclear', 'none', 'n/a', 'on cell type']:
                    continue
                return result
            else:
                result = line.strip()
                # Skip invalid results
                if result.lower() in ['unclear', 'none', 'n/a', 'on cell type']:
                    continue
                return result
    
    return None

def interactive_consensus_annotation(
    marker_genes: Dict[str, List[str]],
    species: str,
    models: List[str] = ['gpt-4o', 'claude-3-opus', 'gemini-2.0-pro'],
    api_keys: Optional[Dict[str, str]] = None,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    max_discussion_rounds: int = 3,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform consensus annotation of cell types using multiple LLMs and interactive resolution.
    
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
        
    Returns:
        Dict[str, Any]: Dictionary containing consensus results and metadata
    """
    from .functions import get_provider
    from .annotate import annotate_clusters
    from .utils import combine_results
    
    # Set up logging
    if verbose:
        write_log("Starting interactive consensus annotation")
    
    # Make sure we have API keys
    if api_keys is None:
        api_keys = {}
        for model in models:
            provider = get_provider(model)
            if provider not in api_keys:
                from .utils import load_api_key
                api_key = load_api_key(provider)
                if api_key:
                    api_keys[provider] = api_key
    
    # Run initial annotations with all models
    model_results = {}
    
    for model in models:
        provider = get_provider(model)
        api_key = api_keys.get(provider)
        
        if not api_key:
            write_log(f"Warning: No API key found for {provider}, skipping {model}", level='warning')
            continue
        
        if verbose:
            write_log(f"Annotating with {model}")
        
        try:
            results = annotate_clusters(
                marker_genes=marker_genes,
                species=species,
                provider=provider,
                model=model,
                api_key=api_key,
                tissue=tissue,
                additional_context=additional_context,
                use_cache=use_cache,
                cache_dir=cache_dir
            )
            
            model_results[model] = results
            
            if verbose:
                write_log(f"Successfully annotated with {model}")
        except Exception as e:
            write_log(f"Error annotating with {model}: {str(e)}", level='error')
    
    # Check if we have any results
    if not model_results:
        write_log("No annotations were successful", level='error')
        return {"error": "No annotations were successful"}
    
    # Check consensus
    consensus, consensus_proportion, entropy, controversial = check_consensus(
        model_results, 
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        api_keys=api_keys
    )
    
    if verbose:
        write_log(f"Found {len(controversial)} controversial clusters out of {len(consensus)}")
        
    # If there are controversial clusters, resolve them
    resolved = {}
    if controversial:
        # Choose best model for discussion
        discussion_model = None
        discussion_provider = None
        
        # Try to use the most capable model available
        for preferred_model in ['gpt-4o', 'claude-3-opus', 'gemini-2.0-pro']:
            if preferred_model in models:
                provider = get_provider(preferred_model)
                if provider in api_keys:
                    discussion_model = preferred_model
                    discussion_provider = provider
                    break
        
        # If no preferred model is available, use the first one
        if not discussion_model and models:
            discussion_model = models[0]
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
                    cache_dir=cache_dir
                )
                
                # Update consensus proportion and entropy for resolved clusters
                for cluster_id, cp in updated_cp.items():
                    consensus_proportion[cluster_id] = cp
                
                for cluster_id, h in updated_h.items():
                    entropy[cluster_id] = h
                
                if verbose:
                    write_log(f"Successfully resolved {len(resolved)} controversial clusters")
            except Exception as e:
                write_log(f"Error resolving controversial clusters: {str(e)}", level='error')
    
    # Merge consensus and resolved
    final_annotations = consensus.copy()
    for cluster_id, annotation in resolved.items():
        final_annotations[cluster_id] = annotation
    
    # Clean all annotations, ensure special markers are removed
    cleaned_annotations = {}
    for cluster_id, annotation in final_annotations.items():
        cleaned_annotations[cluster_id] = clean_annotation(annotation)
    
    # Prepare results
    result = {
        "consensus": cleaned_annotations,
        "consensus_proportion": consensus_proportion,
        "entropy": entropy,
        "controversial_clusters": controversial,
        "resolved": resolved,
        "model_annotations": model_results,
        "discussion_logs": discussion_logs if 'discussion_logs' in locals() else {},
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": models,
            "species": species,
            "tissue": tissue,
            "consensus_threshold": consensus_threshold,
            "entropy_threshold": entropy_threshold,
            "max_discussion_rounds": max_discussion_rounds
        }
    }
    
    return result

def print_consensus_summary(result: Dict[str, Any]) -> None:
    """
    Print a summary of consensus annotation results.
    
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
    if metadata.get('tissue'):
        print(f"Tissue: {metadata['tissue']}")
    print(f"Models used: {', '.join(metadata.get('models', []))}")
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
                if isinstance(logs, list):
                    # If it's a list, join it into a string
                    logs_text = "\n".join(logs)
                else:
                    # If it's already a string, use it directly
                    logs_text = logs
                    
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
    marker_genes: List[str],
    model_votes: Dict[str, str],
    species: str,
    tissue: Optional[str] = None,
    provider: str = 'openai',
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True
) -> str:
    """
    Facilitate a discussion between different model predictions for a controversial cluster.
    
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
    from .annotate import get_model_response
    
    # Generate discussion prompt
    prompt = create_discussion_prompt(
        cluster_id=cluster_id,
        marker_genes=marker_genes,
        model_votes=model_votes,
        species=species,
        tissue=tissue
    )
    
    # Get response
    response = get_model_response(prompt, provider, model, api_key, use_cache)
    
    # Extract final decision
    cell_type = extract_cell_type_from_discussion(response)
    
    # Return the full discussion and the extracted cell type
    return f"{response}\n\nFINAL DETERMINATION: {cell_type}"

def summarize_discussion(discussion: str) -> str:
    """
    Summarize a model discussion about cell type annotation.
    
    Args:
        discussion: Full discussion text
        
    Returns:
        str: Summary of the discussion
    """
    # Extract key points from the discussion
    lines = discussion.strip().split('\n')
    summary_lines = []
    
    # Look for common summary indicators
    for line in lines:
        line = line.strip()
        if (line.lower().startswith(("conclusion", "summary", "final", "therefore", "overall", "in summary"))):
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