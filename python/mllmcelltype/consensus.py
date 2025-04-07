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

def check_consensus(predictions: Dict[str, Dict[str, str]], 
                  threshold: float = 0.6) -> Tuple[Dict[str, str], Dict[str, float], List[str]]:
    """
    Check if there is consensus among different model predictions.
    
    Args:
        predictions: Dictionary mapping model names to dictionaries of cluster annotations
        threshold: Agreement threshold below which a cluster is considered controversial
        
    Returns:
        Tuple of:
            - Dictionary mapping cluster IDs to consensus annotations
            - Dictionary mapping cluster IDs to confidence scores
            - List of controversial cluster IDs
    """
    from .utils import find_agreement
    
    # Find consensus annotations and confidence scores
    consensus, confidence = find_agreement(predictions)
    
    # Find controversial clusters
    controversial = [cluster for cluster, score in confidence.items() if score < threshold]
    
    return consensus, confidence, controversial

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
    use_cache: bool = True,
    cache_dir: Optional[str] = None
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
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
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        
    Returns:
        Tuple[Dict[str, str], Dict[str, List[str]]]: 
            - Dictionary mapping cluster IDs to resolved annotations
            - Dictionary mapping cluster IDs to discussion history for each round
    """
    from .annotate import get_model_response
    from .prompts import create_consensus_check_prompt
    import math
    from collections import Counter
    
    results = {}
    discussion_history = {}
    
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
    
        # Calculate initial consensus metrics
        vote_counts = Counter(current_votes.values())
        total_votes = len(current_votes)
        if total_votes > 0:
            most_common_vote, most_common_count = vote_counts.most_common(1)[0]
            cp = most_common_count / total_votes  # Consensus Proportion
            
            # Calculate Shannon Entropy
            h = 0
            for vote, count in vote_counts.items():
                p = count / total_votes
                h -= p * math.log2(p)
            
            write_log(f"Initial metrics for cluster {cluster_id}: CP={cp:.2f}, H={h:.2f}")
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
                        
                        # 添加更严格的判断逻辑，要求至少包含两个共识指标
                        indicators_found = sum(1 for indicator in consensus_indicators if indicator in consensus_response.lower())
                        
                        # 在第一轮讨论中，要求更严格的条件
                        if current_round == 1:
                            consensus_reached = indicators_found >= 2 and "strong evidence" in consensus_response.lower()
                        else:
                            consensus_reached = indicators_found >= 1
                        
                        if consensus_reached:
                            final_decision = round_decision
                            write_log(f"Consensus reached for cluster {cluster_id} in round {current_round}", level='info')
                            
                            # Calculate final consensus metrics
                            # For simplicity, we'll set CP=1.0 and H=0.0 when consensus is explicitly reached
                            rounds_history.append(f"Consensus reached in round {current_round}\nFinal cell type: {final_decision}\nConsensus Proportion (CP): 1.00\nShannon Entropy (H): 0.00")
                
                    # Move to next round if no consensus yet
                    if not consensus_reached:
                        current_round += 1
                
                # After all rounds, use the last round's decision if no consensus was explicitly reached
                if not final_decision and round_decision:
                    final_decision = round_decision
                    write_log(f"Using final round decision for cluster {cluster_id} after {max_discussion_rounds} rounds", level='info')
                
                # Store the final result
                if not final_decision:
                    write_log(f"Warning: Could not reach a decision for cluster {cluster_id} after {max_discussion_rounds} rounds", level='warning')
                    results[cluster_id] = "Inconclusive"
                else:
                    results[cluster_id] = final_decision
                
                # Store the full discussion history
                discussion_history[cluster_id] = rounds_history
                    
            except Exception as e:
                write_log(f"Error during discussion for cluster {cluster_id}: {str(e)}", level='error')
                results[cluster_id] = f"Error during discussion: {str(e)}"
                discussion_history[cluster_id] = [f"Error occurred: {str(e)}"]
    
    return results, discussion_history

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
    consensus_threshold: float = 0.6,
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
    consensus, confidence, controversial = check_consensus(model_results, consensus_threshold)
    
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
                resolved, discussion_logs = process_controversial_clusters(
                    marker_genes=marker_genes,
                    controversial_clusters=controversial,
                    model_predictions=model_results,
                    species=species,
                    tissue=tissue,
                    provider=discussion_provider,
                    model=discussion_model,
                    api_key=api_keys.get(discussion_provider),
                    max_discussion_rounds=max_discussion_rounds,
                    use_cache=use_cache,
                    cache_dir=cache_dir
                )
                
                if verbose:
                    write_log(f"Successfully resolved {len(resolved)} controversial clusters")
            except Exception as e:
                write_log(f"Error resolving controversial clusters: {str(e)}", level='error')
    
    # Merge consensus and resolved
    final_annotations = consensus.copy()
    for cluster_id, annotation in resolved.items():
        final_annotations[cluster_id] = annotation
    
    # Prepare results
    result = {
        "consensus": final_annotations,
        "confidence": confidence,
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
    
    # Print consensus annotations with confidence
    consensus = result.get("consensus", {})
    confidence = result.get("confidence", {})
    resolved = result.get("resolved", {})
    
    print("Cluster annotations:")
    for cluster, annotation in sorted(consensus.items(), key=lambda x: x[0]):
        conf = confidence.get(cluster, 0)
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
            # For non-resolved clusters, calculate CP directly (same as old confidence)
            # and set H to 0 for perfect agreement or a higher value for disagreement
            cp_value = conf  # CP is equivalent to the old confidence metric
            
            # Calculate H for non-resolved clusters
            # If all models agree (conf=1.0), H=0
            # If there's disagreement, calculate a simple entropy value
            if conf == 1.0:
                h_value = 0.0
            else:
                # Simple entropy calculation based on agreement level
                # Less agreement = higher entropy
                h_value = -1 * (conf * math.log2(conf) + (1-conf) * math.log2(1-conf) if conf > 0 else 0)
                
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