#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic usage example for LLMCelltype.
This script demonstrates how to use LLMCelltype to annotate cell clusters.
"""

import os
import sys
import pandas as pd

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mllmcelltype import annotate_clusters, setup_logging, load_api_key

# Setup logging
setup_logging(log_level="INFO")

# Example marker genes data
# In a real scenario, this would be loaded from a file or generated from analysis
marker_genes_data = {
    "cluster": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "gene": ["CD3D", "CD3E", "CD2", "CD19", "MS4A1", "CD79A", "FCGR3A", "CD14", "LYZ"],
    "avg_log2FC": [2.5, 2.3, 2.1, 3.0, 2.8, 2.7, 2.2, 2.0, 1.9],
    "pct.1": [0.9, 0.85, 0.8, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    "pct.2": [0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "p_val_adj": [1e-10, 1e-9, 1e-8, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
}

marker_genes_df = pd.DataFrame(marker_genes_data)

def main():
    """Main function to demonstrate LLMCelltype usage."""
    print("LLMCelltype Basic Usage Example")
    print("-------------------------------")
    
    # Load API key from environment variable or .env file
    api_key = load_api_key("openai")
    
    if not api_key:
        print("Warning: OpenAI API key not found. Using placeholder for demonstration.")
        api_key = "your-openai-api-key"  # This won't work for actual API calls
    
    # Print marker genes
    print("\nMarker Genes:")
    print(marker_genes_df)
    
    # Annotate clusters
    print("\nAnnotating clusters...")
    try:
        annotations = annotate_clusters(
            marker_genes=marker_genes_df,
            species="human",
            provider="openai",
            model="gpt-4o",
            tissue="blood",
            api_key=api_key
        )
        
        # Print annotations
        print("\nAnnotation Results:")
        for cluster, annotation in annotations.items():
            print(f"Cluster {cluster}: {annotation}")
            
    except Exception as e:
        print(f"Error during annotation: {e}")
        print("Note: This example requires a valid OpenAI API key to run.")
        print("You can replace 'openai' with other providers like 'anthropic', 'gemini', etc.")
    
    print("\nExample completed.")

if __name__ == "__main__":
    main()
