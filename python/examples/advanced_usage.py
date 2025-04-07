#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced usage example for LLMCelltype.
This script demonstrates batch annotation and multiple provider usage.
"""

import os
import pandas as pd
from mllmcelltype import batch_annotate_clusters, setup_logging, load_api_key

# Setup logging with file output
setup_logging(log_level="INFO", log_dir="./logs")

# Example marker genes for multiple clusters
def create_example_marker_genes(cluster_ids, gene_sets):
    """Create example marker genes dataframe for demonstration."""
    data = {
        "cluster": [],
        "gene": [],
        "avg_log2FC": [],
        "pct.1": [],
        "pct.2": [],
        "p_val_adj": []
    }
    
    for cluster_id, genes in zip(cluster_ids, gene_sets):
        for i, gene in enumerate(genes):
            data["cluster"].append(cluster_id)
            data["gene"].append(gene)
            data["avg_log2FC"].append(2.0 + i * 0.1)
            data["pct.1"].append(0.9 - i * 0.05)
            data["pct.2"].append(0.1 + i * 0.05)
            data["p_val_adj"].append(1e-10 * (10 ** i))
    
    return pd.DataFrame(data)

# Example datasets
def create_example_datasets():
    """Create example datasets for demonstration."""
    # Dataset 1: Immune cells
    immune_clusters = [1, 2, 3]
    immune_genes = [
        ["CD3D", "CD3E", "CD2"],      # T cells
        ["CD19", "MS4A1", "CD79A"],   # B cells
        ["FCGR3A", "CD14", "LYZ"]     # Monocytes
    ]
    
    # Dataset 2: Brain cells
    brain_clusters = [1, 2, 3, 4]
    brain_genes = [
        ["SLC17A7", "SATB2", "CAMK2A"],  # Excitatory neurons
        ["GAD1", "GAD2", "SLC32A1"],     # Inhibitory neurons
        ["AQP4", "GFAP", "SLC1A3"],      # Astrocytes
        ["MOG", "MBP", "PLP1"]           # Oligodendrocytes
    ]
    
    # Dataset 3: Liver cells
    liver_clusters = [1, 2, 3]
    liver_genes = [
        ["ALB", "APOA1", "APOB"],      # Hepatocytes
        ["PECAM1", "VWF", "CDH5"],     # Endothelial cells
        ["LRAT", "RBP1", "GFAP"]       # Stellate cells
    ]
    
    return [
        create_example_marker_genes(immune_clusters, immune_genes),
        create_example_marker_genes(brain_clusters, brain_genes),
        create_example_marker_genes(liver_clusters, liver_genes)
    ]

def compare_providers(marker_genes_df, species, tissue, providers_models):
    """Compare annotations from different providers."""
    results = {}
    
    for provider, model in providers_models.items():
        print(f"\nTrying provider: {provider} with model: {model}")
        
        # Load API key
        api_key_env = f"{provider.upper()}_API_KEY"
        api_key = load_api_key(api_key_env)
        
        if not api_key:
            print(f"Warning: {api_key_env} not found. Skipping {provider}.")
            continue
        
        try:
            # Annotate with current provider
            annotations = batch_annotate_clusters(
                marker_genes_list=[marker_genes_df],
                species=species,
                provider=provider,
                model=model,
                tissue=tissue,
                api_key=api_key,
                max_retries=2,
                timeout=30
            )[0]  # Get first result since we only passed one dataset
            
            results[provider] = annotations
            print(f"✓ {provider} annotation successful")
            
        except Exception as e:
            print(f"✗ Error with {provider}: {e}")
    
    return results

def main():
    """Main function to demonstrate advanced LLMCelltype usage."""
    print("LLMCelltype Advanced Usage Example")
    print("---------------------------------")
    
    # Create example datasets
    datasets = create_example_datasets()
    tissues = ["blood", "brain", "liver"]
    
    # Define providers and models to try
    providers_models = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-opus-20240229",
        "gemini": "gemini-1.5-pro",
        "deepseek": "deepseek-chat",
        "zhipu": "glm-4"
    }
    
    print("\nBatch Annotation Example:")
    print("------------------------")
    
    # Try to load at least one API key
    has_api_key = False
    for provider in providers_models:
        if load_api_key(provider):
            has_api_key = True
            break
    
    if not has_api_key:
        print("Warning: No API keys found. Using demonstration mode.")
        print("In a real scenario, you would need valid API keys.")
    
    try:
        # Example 1: Batch annotation
        print("\nPerforming batch annotation across multiple datasets...")
        
        # Use the first available provider
        for provider, model in providers_models.items():
            api_key = load_api_key(provider)
            if api_key or not has_api_key:
                print(f"Using provider: {provider}")
                
                # Batch annotate all datasets
                batch_results = batch_annotate_clusters(
                    marker_genes_list=datasets,
                    species="human",  # Changed from list to string
                    tissue=tissues,
                    provider=provider,
                    model=model,
                    api_key=api_key if has_api_key else "demo-key"
                    # Note: dry_run parameter was removed as it's not supported
                )
                
                # Print results
                for i, (result, tissue) in enumerate(zip(batch_results, tissues)):
                    print(f"\nDataset {i+1} ({tissue}):")
                    for cluster, annotation in result.items():
                        print(f"  Cluster {cluster}: {annotation}")
                
                break
        
        # Example 2: Provider comparison
        print("\n\nProvider Comparison Example:")
        print("--------------------------")
        print("Comparing annotations from different providers for brain cells...")
        
        if has_api_key:
            comparison_results = compare_providers(
                marker_genes_df=datasets[1],  # Brain cells dataset
                species="human",
                tissue="brain",
                providers_models=providers_models
            )
            
            # Print comparison
            if comparison_results:
                print("\nComparison Results:")
                clusters = sorted(list(next(iter(comparison_results.values())).keys()))
                
                # Header
                print(f"{'Cluster':<8}", end="")
                for provider in comparison_results:
                    print(f"{provider:<15}", end="")
                print()
                
                # Print a line for each cluster
                for cluster in clusters:
                    print(f"{cluster:<8}", end="")
                    for provider, annotations in comparison_results.items():
                        annotation = annotations.get(cluster, "N/A")
                        # Truncate long annotations
                        if len(annotation) > 12:
                            annotation = annotation[:12] + "..."
                        print(f"{annotation:<15}", end="")
                    print()
        else:
            print("Skipping provider comparison (no API keys available)")
    
    except Exception as e:
        print(f"Error in advanced example: {e}")
    
    print("\nAdvanced example completed.")

if __name__ == "__main__":
    main()
