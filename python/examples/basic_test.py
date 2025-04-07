#!/usr/bin/env python3
"""
Basic test script for LLMCelltype functionality.
"""

import os
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import llmcelltype as lct

# Load API keys from .env file
load_dotenv()

# Sample data - common cell types in human PBMCs
marker_genes = {
    "1": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7", "LCK", "CD8A"],
    "2": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74", "CD22"],
    "3": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A", "S100A8", "S100A9"],
    "4": ["FCGR3A", "NCAM1", "KLRB1", "GZMB", "NKG7", "CD160", "FCER1G", "KLRC1", "CD56"],
    "5": ["HBA1", "HBA2", "HBB", "ALAS2", "AHSP", "CA1", "HEMGN"]
}

def test_provider(provider, model):
    """Test a specific provider"""
    print(f"\nTesting {provider} with model {model}...")
    
    # Get API key
    api_key = lct.load_api_key(provider)
    if not api_key:
        print(f"Error: {provider} API key not found.")
        return None
        
    try:
        # Annotate cell types
        results = lct.annotate_clusters(
            marker_genes=marker_genes,
            species="human",
            tissue="peripheral blood",
            provider=provider,
            model=model,
            api_key=api_key,
            use_cache=True
        )
        
        print(f"\n{provider} annotation results:")
        for cluster, cell_type in results.items():
            print(f"Cluster {cluster}: {cell_type}")
            
        return results
    except Exception as e:
        print(f"Error testing {provider}: {str(e)}")
        return None

def run_test(provider=None):
    """Run a basic test of LLMCelltype functionality"""
    print(f"LLMCelltype version: {lct.__version__}")
    
    # Setup logging
    lct.setup_logging(log_level='INFO')
    lct.write_log("Starting basic test")
    
    # Define provider-model pairs to test
    provider_models = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20240620",
        "gemini": "gemini-1.5-pro"
    }
    
    # Test specific provider if requested
    if provider and provider in provider_models:
        result = test_provider(provider, provider_models[provider])
        return
    
    # Otherwise test all available providers
    results_by_model = {}
    
    for provider, model in provider_models.items():
        result = test_provider(provider, model)
        if result:
            results_by_model[f"{provider} ({model})"] = result
    
    # Compare results if we have at least 2 models
    if len(results_by_model) >= 2:
        print("\n=== Model Comparison ===\n")
        
        # Compare model predictions
        agreement_df, metrics = lct.compare_model_predictions(
            model_predictions=results_by_model,
            display_plot=False
        )
        
        print(f"Average agreement: {metrics['agreement_avg']:.2f}")
        print(f"Most agreeing pair: {metrics['most_agreeing_pair'][0]} and {metrics['most_agreeing_pair'][1]} ({metrics['most_agreeing_score']:.2f})")
        print(f"Least agreeing pair: {metrics['least_agreeing_pair'][0]} and {metrics['least_agreeing_pair'][1]} ({metrics['least_agreeing_score']:.2f})")
        
        # Print comparison table
        comparison_table = lct.create_comparison_table(results_by_model)
        print("\nComparison Table:")
        print(comparison_table.to_string())
        
        # Check consensus
        consensus, confidence, controversial = lct.check_consensus(results_by_model)
        
        print("\nConsensus annotations:")
        for cluster, anno in consensus.items():
            print(f"Cluster {cluster}: {anno} (Confidence: {confidence.get(cluster, 0):.2f})")
        
        if controversial:
            print(f"\nFound {len(controversial)} controversial clusters: {', '.join(controversial)}")
        else:
            print("\nNo controversial clusters found!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLMCelltype functionality")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], 
                        help="Specific provider to test")
    args = parser.parse_args()
    
    run_test(args.provider)