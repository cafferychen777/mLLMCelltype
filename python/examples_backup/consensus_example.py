#!/usr/bin/env python3
"""
Example demonstrating consensus annotation using LLMCelltype.
"""

import os
import sys
import pandas as pd
import json
from dotenv import load_dotenv

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mllmcelltype as lct

# Load API keys from .env file
load_dotenv()

# Sample data
marker_genes = {
    "1": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7", "LCK", "CD8A"],
    "2": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74", "CD22"],
    "3": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A", "S100A8", "S100A9"],
    "4": ["FCGR3A", "NCAM1", "KLRB1", "GZMB", "NKG7", "CD160", "FCER1G", "KLRC1", "CD56"],
    "5": ["HBA1", "HBA2", "HBB", "ALAS2", "AHSP", "CA1", "HEMGN"]
}

def run_consensus_example():
    """Run the consensus annotation example"""
    print("\n=== Consensus Annotation Example ===\n")
    
    # Setup logging
    lct.setup_logging(log_level='INFO')
    
    # Get available API keys
    api_keys = {}
    for provider in ['openai', 'anthropic', 'gemini']:
        api_key = lct.load_api_key(provider)
        if api_key:
            api_keys[provider] = api_key
    
    if not api_keys:
        print("Error: No API keys found. Please set at least one API key in your .env file.")
        return
    
    # Define models to use
    models = []
    if 'openai' in api_keys:
        models.append('gpt-4o')
    if 'anthropic' in api_keys:
        models.append('claude-3-5-sonnet-20241022')  # Latest Claude model
    if 'gemini' in api_keys:
        models.append('gemini-1.5-pro')  # Gemini model name without models/ prefix
    if 'qwen' in api_keys:
        models.append('qwen2.5-72b-instruct')  # Alibaba's Qwen model
    
    if len(models) < 2:
        print(f"Warning: Consensus annotation works best with at least 2 models, but only {len(models)} found.")
    
    print(f"Using models: {', '.join(models)}")
    
    # Run interactive consensus annotation with iterative discussion
    result = lct.interactive_consensus_annotation(
        marker_genes=marker_genes,
        species='human',
        tissue='peripheral blood',
        models=models,
        api_keys=api_keys,
        consensus_threshold=0.6,
        max_discussion_rounds=3,  # Enable up to 3 rounds of iterative discussion
        use_cache=True,
        verbose=True
    )
    
    # Print consensus summary
    lct.print_consensus_summary(result)
    
    # Print discussion logs for controversial clusters
    if "discussion_logs" in result and result["discussion_logs"]:
        print("\n=== Discussion Logs for Controversial Clusters ===\n")
        for cluster_id, logs in result["discussion_logs"].items():
            print(f"Cluster {cluster_id} Discussion:")
            print(f"{'='*50}")
            print(logs)
            print(f"{'='*50}\n")
    
    # Create comparison table
    model_annotations = result.get("model_annotations", {})
    if len(model_annotations) >= 2:
        print("\n=== Model Comparison ===\n")
        
        # Compare model predictions
        agreement_df, metrics = lct.compare_model_predictions(
            model_predictions=model_annotations,
            display_plot=False
        )
        
        print(f"Average agreement: {metrics['agreement_avg']:.2f}")
        print(f"Most agreeing pair: {metrics['most_agreeing_pair'][0]} and {metrics['most_agreeing_pair'][1]} ({metrics['most_agreeing_score']:.2f})")
        print(f"Least agreeing pair: {metrics['least_agreeing_pair'][0]} and {metrics['least_agreeing_pair'][1]} ({metrics['least_agreeing_score']:.2f})")
        
        # Create comparison table
        comparison_table = lct.create_comparison_table(model_annotations)
        print("\nComparison Table:")
        print(comparison_table.to_string(index=False))
        
        # Analyze confusion patterns
        confusion = lct.analyze_confusion_patterns(model_annotations)
        if "common_disagreement_pairs" in confusion and confusion["common_disagreement_pairs"]:
            print("\nCommon Disagreement Patterns:")
            for pair, count in confusion["common_disagreement_pairs"][:3]:
                print(f"  - '{pair[0]}' vs '{pair[1]}': {count} occurrences")

if __name__ == "__main__":
    run_consensus_example()