#!/usr/bin/env python
"""
Test script for LLM consensus annotation functionality.
"""

import os
import sys
import json
from typing import Dict, List

# Add the package directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from mllmcelltype.consensus import check_consensus_with_llm, check_consensus
from mllmcelltype.prompts import create_consensus_check_prompt
from mllmcelltype.utils import find_agreement, clean_annotation

# Sample data for testing
test_predictions = {
    "model1": {
        "cluster1": "T cells",
        "cluster2": "B cells",
        "cluster3": "Macrophages"
    },
    "model2": {
        "cluster1": "CD4+ T cells",
        "cluster2": "B lymphocytes",
        "cluster3": "Monocytes"
    },
    "model3": {
        "cluster1": "T lymphocytes",
        "cluster2": "B cells",
        "cluster3": "Tissue-resident macrophages"
    }
}

# Test with different levels of agreement
test_predictions_with_disagreement = {
    "model1": {
        "cluster1": "T cells",
        "cluster2": "B cells",
        "cluster3": "Macrophages",
        "cluster4": "Dendritic cells"
    },
    "model2": {
        "cluster1": "CD4+ T cells",
        "cluster2": "B lymphocytes",
        "cluster3": "Monocytes",
        "cluster4": "Plasmacytoid dendritic cells"
    },
    "model3": {
        "cluster1": "T lymphocytes",
        "cluster2": "B cells",
        "cluster3": "Tissue-resident macrophages",
        "cluster4": "Natural killer cells"
    }
}

def print_results(title: str, consensus: Dict[str, str], 
                 consensus_proportion: Dict[str, float], 
                 entropy: Dict[str, float]) -> None:
    """Print the results in a formatted way."""
    print(f"\n=== {title} ===")
    print(f"{'Cluster':<10} {'Consensus':<30} {'Proportion':<10} {'Entropy':<10}")
    print("-" * 60)
    
    for cluster in sorted(consensus.keys()):
        print(f"{cluster:<10} {consensus[cluster]:<30} {consensus_proportion[cluster]:.2f}      {entropy[cluster]:.2f}")

def main():
    print("Testing LLM Consensus Annotation")
    print("--------------------------------")
    
    # Create a custom simple version of the consensus check prompt function
    def simple_consensus_check_prompt(annotations):
        prompt = """You are an expert in single-cell RNA-seq analysis and cell type annotation.

I need you to analyze the following cell type annotations from different models for the same cluster and determine if there is a consensus.

The annotations are:
{annotations}

Please analyze these annotations and determine:
1. If there is a consensus (1 for yes, 0 for no)
2. The consensus proportion (between 0 and 1)
3. An entropy value measuring the diversity of opinions (higher means more diverse)
4. The best consensus annotation

Respond with exactly 4 lines:
Line 1: 0 or 1 (consensus reached?)
Line 2: Consensus proportion (e.g., 0.75)
Line 3: Entropy value (e.g., 0.85)
Line 4: The consensus cell type (or most likely if no clear consensus)

Only output these 4 lines, nothing else."""
        
        # Format the annotations
        formatted_annotations = "\n".join([f"- {anno}" for anno in annotations])
        
        # Replace the placeholder
        prompt = prompt.replace("{annotations}", formatted_annotations)
        
        return prompt
    
    # Test prompt generation
    test_annotations = ["T cells", "CD4+ T cells", "T lymphocytes"]
    prompt = simple_consensus_check_prompt(test_annotations)
    print("\nGenerated prompt for consensus check:")
    print(prompt)
    
    # Test find_agreement function
    print("\nTesting find_agreement function:")
    consensus, consensus_proportion, entropy = find_agreement(test_predictions)
    print_results("Results from find_agreement", consensus, consensus_proportion, entropy)
    
    # Test check_consensus function
    print("\nTesting check_consensus function:")
    consensus, consensus_proportion, entropy, controversial = check_consensus(test_predictions)
    print_results("Results from check_consensus", consensus, consensus_proportion, entropy)
    
    # Test with disagreement
    print("\nTesting with disagreement:")
    consensus, consensus_proportion, entropy, controversial = check_consensus(test_predictions_with_disagreement)
    print_results("Results with disagreement", consensus, consensus_proportion, entropy)
    print(f"Controversial clusters: {controversial}")
    
    # Test LLM consensus (if API keys are available)
    try:
        print("\nTesting check_consensus_with_llm function:")
        consensus, consensus_proportion, entropy = check_consensus_with_llm(test_predictions)
        print_results("Results from check_consensus_with_llm", consensus, consensus_proportion, entropy)
    except Exception as e:
        print(f"\nError testing check_consensus_with_llm: {str(e)}")
        print("This may be due to missing API keys. Please set the appropriate API keys for Qwen or Claude.")

if __name__ == "__main__":
    main()
