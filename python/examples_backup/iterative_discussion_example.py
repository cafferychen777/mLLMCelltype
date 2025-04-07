#!/usr/bin/env python3
"""
Example demonstrating iterative discussion for controversial clusters using LLMCelltype.
This example focuses on myeloid cells in lung tissue, similar to the framework animation.
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

# Sample data for myeloid cells in lung tissue
# These markers are chosen to create deliberate controversy between macrophages and dendritic cells
marker_genes = {
    "1": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7", "LCK", "CD8A"],  # T cells
    "2": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74", "CD22"],  # B cells
    "3": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "S100A8", "S100A9"],   # Monocytes/Macrophages
    "4": ["CD68", "CD11c", "MARCO", "CD206", "MERTK", "SIGLEC1"],         # Controversial: Macrophages vs DCs
    "5": ["HBA1", "HBA2", "HBB", "ALAS2", "AHSP", "CA1", "HEMGN"]         # Erythrocytes
}

def run_iterative_discussion_example():
    """Run the iterative discussion example for controversial clusters"""
    print("\n=== Iterative Discussion Example for Controversial Clusters ===\n")
    
    # Setup logging
    lct.setup_logging(log_level='INFO')
    
    # Get available API keys
    api_keys = {}
    for provider in ['openai', 'anthropic', 'gemini', 'qwen']:
        api_key = lct.load_api_key(provider)
        if api_key:
            api_keys[provider] = api_key
    
    if not api_keys:
        print("Error: No API keys found. Please set at least one API key in your .env file.")
        return
    
    # Define models to use - deliberately use models that might disagree
    models = []
    if 'openai' in api_keys:
        models.append('gpt-4o')
    if 'anthropic' in api_keys:
        models.append('claude-3-5-sonnet-20241022')
    if 'gemini' in api_keys:
        models.append('gemini-1.5-pro')
    if 'qwen' in api_keys:
        models.append('qwen2.5-72b-instruct')
    
    if len(models) < 2:
        print(f"Warning: Consensus annotation works best with at least 2 models, but only {len(models)} found.")
        return
    
    print(f"Using models: {', '.join(models)}")
    
    # First run with only 1 discussion round to show initial disagreement
    print("\n=== First Run: Single Discussion Round ===\n")
    result_single = lct.interactive_consensus_annotation(
        marker_genes=marker_genes,
        species='human',
        tissue='lung',
        models=models,
        api_keys=api_keys,
        consensus_threshold=0.6,
        max_discussion_rounds=1,  # Only 1 round
        use_cache=True,
        verbose=True
    )
    
    # Print summary of single round discussion
    lct.print_consensus_summary(result_single)
    
    # Now run with multiple discussion rounds
    print("\n=== Second Run: Multiple Discussion Rounds ===\n")
    result_multi = lct.interactive_consensus_annotation(
        marker_genes=marker_genes,
        species='human',
        tissue='lung',
        models=models,
        api_keys=api_keys,
        consensus_threshold=0.9,  # Increase consensus threshold to make it harder to reach consensus
        max_discussion_rounds=3,  # Up to 3 rounds
        use_cache=True,
        verbose=True
    )
    
    # Print summary of multi-round discussion
    lct.print_consensus_summary(result_multi)
    
    # Print detailed discussion logs for the controversial cluster #4
    if "discussion_logs" in result_multi and "4" in result_multi["discussion_logs"]:
        print("\n=== Detailed Discussion Logs for Cluster #4 (Myeloid Cells) ===\n")
        print(f"{'='*80}")
        print(result_multi["discussion_logs"]["4"])
        print(f"{'='*80}\n")
        
        # Extract and display consensus metrics evolution
        print("\n=== Consensus Metrics Evolution for Cluster #4 ===\n")
        logs = result_multi["discussion_logs"]["4"]
        
        # Convert logs to string if it's a list
        if isinstance(logs, list):
            logs_text = "\n".join(logs)
        else:
            logs_text = logs
            
        rounds = logs_text.split("Round ")
        
        print("| Round | Proposed Cell Type | CP Value | H Value |")
        print("|-------|-------------------|----------|---------|")
        
        # Extract initial metrics if available
        if "Initial votes" in logs_text:
            initial_section = logs_text.split("Round ")[0]
            cp_value = "N/A"
            h_value = "N/A"
            cell_type = "N/A"
            
            for line in initial_section.split("\n"):
                if "Consensus Proportion (CP):" in line:
                    cp_parts = line.split("Consensus Proportion (CP):")[1].strip().split()
                    if cp_parts:
                        cp_value = cp_parts[0]
                if "Shannon Entropy (H):" in line:
                    h_parts = line.split("Shannon Entropy (H):")[1].strip().split()
                    if h_parts:
                        h_value = h_parts[0]
            
            print(f"| Initial | {cell_type} | {cp_value} | {h_value} |")
        
        # Extract metrics from each round
        for i in range(1, 4):  # Check rounds 1-3
            round_marker = f"{i} Discussion:"
            if round_marker in logs_text:
                round_section = logs_text.split(round_marker)[1].split("Round ")[0] if i < 3 else logs_text.split(round_marker)[1]
                cp_value = "N/A"
                h_value = "N/A"
                cell_type = "N/A"
                
                # Find proposed cell type
                for line in round_section.split("\n"):
                    if "Proposed cell type:" in line:
                        cell_parts = line.split("Proposed cell type:")[1].strip()
                        if cell_parts:
                            cell_type = cell_parts
                
                # Find metrics
                for line in round_section.split("\n"):
                    if "Consensus Proportion (CP):" in line:
                        cp_parts = line.split("Consensus Proportion (CP):")[1].strip().split()
                        if cp_parts:
                            cp_value = cp_parts[0]
                    if "Shannon Entropy (H):" in line:
                        h_parts = line.split("Shannon Entropy (H):")[1].strip().split()
                        if h_parts:
                            h_value = h_parts[0]
                
                print(f"| {i} | {cell_type} | {cp_value} | {h_value} |")
    
    # Compare the final results between single and multiple rounds
    print("\n=== Comparison: Single Round vs Multiple Rounds ===\n")
    single_resolved = result_single.get("resolved", {})
    multi_resolved = result_multi.get("resolved", {})
    
    print("| Cluster | Single Round | Multiple Rounds |")
    print("|---------|--------------|----------------|")
    
    for cluster in marker_genes.keys():
        if cluster in single_resolved or cluster in multi_resolved:
            single_result = single_resolved.get(cluster, "Not resolved")
            multi_result = multi_resolved.get(cluster, "Not resolved")
            print(f"| {cluster} | {single_result} | {multi_result} |")

if __name__ == "__main__":
    run_iterative_discussion_example()
