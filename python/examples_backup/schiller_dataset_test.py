#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script: Using the Schiller_2020.h5ad dataset to test the iterative discussion functionality of LLMCelltype
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from dotenv import load_dotenv

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions from LLMCelltype
from mllmcelltype import interactive_consensus_annotation, print_consensus_summary

# Load environment variables
load_dotenv()

def run_schiller_dataset_test():
    """Test the iterative discussion functionality of LLMCelltype using the Schiller_2020 dataset"""
    
    print("\n=== Schiller 2020 Dataset Test with Iterative Discussion ===\n")
    
    # Load dataset
    print("Loading Schiller 2020 dataset...")
    data_path = "/Users/apple/Research/LLMCelltype/data/raw/Schiller_2020.h5ad"
    adata = sc.read_h5ad(data_path)
    
    print(f"Dataset loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Check if clustering results already exist in the dataset
    if 'leiden' not in adata.obs.columns:
        print("\nPreprocessing dataset...")
        # Normalize data
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Select highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        print(f"Selected {sum(adata.var.highly_variable)} highly variable genes")
        
        # Principal component analysis
        sc.pp.pca(adata, svd_solver='arpack')
        
        # Calculate neighbor graph
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        
        # Cluster using Leiden algorithm
        print("Performing clustering...")
        sc.tl.leiden(adata, resolution=0.8)
        print(f"Found {len(adata.obs['leiden'].unique())} clusters")
        
        # Calculate UMAP embeddings for visualization
        sc.tl.umap(adata)
        
        # Select a subset of cells for annotation (to speed up processing)
        print("Sampling cells for annotation...")
        # Randomly select some cells from each cluster
        np.random.seed(42)  # Set random seed to ensure reproducibility
        
        # If there are too many clusters, only select the first 10 clusters for testing
        clusters_to_use = sorted(adata.obs['leiden'].unique())[:10]
        print(f"Using clusters: {', '.join(clusters_to_use)}")
        
        # Create a new AnnData object containing only the selected clusters
        adata_subset = adata[adata.obs['leiden'].isin(clusters_to_use)].copy()
        print(f"Subset created with {adata_subset.shape[0]} cells from {len(clusters_to_use)} clusters")
        
        # Use the subset for subsequent analysis
        adata = adata_subset
    
    # Get API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'qwen': os.getenv('QWEN_API_KEY')
    }
    
    # Models to use
    models = ['gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro', 'qwen2.5-72b-instruct']
    print(f"Using models: {', '.join(models)}")
    
    # No need to create an instance, use the function directly
    
    # Extract marker genes
    print("\nExtracting marker genes...")
    
    # Create mapping from gene IDs to gene names
    gene_id_to_name = dict(zip(adata.var.index, adata.var['original_gene_symbols']))
    
    # Calculate marker genes for each cluster
    print("Computing marker genes...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    # Extract top 10 marker genes for each cluster and convert to gene names
    marker_genes = {}
    for cluster in adata.obs['leiden'].unique():
        try:
            # Get gene IDs
            gene_ids = sc.get.rank_genes_groups_df(adata, group=cluster).head(10)['names'].tolist()
            # Convert gene IDs to gene names
            gene_names = [gene_id_to_name.get(gene_id, gene_id) for gene_id in gene_ids]
            marker_genes[cluster] = gene_names
        except Exception as e:
            print(f"Error extracting marker genes for cluster {cluster}: {e}")
            # If an error occurs, use some basic marker genes
            marker_genes[cluster] = ['CD3D', 'CD3E', 'CD4', 'CD8A', 'MS4A1', 'CD19', 'CD14', 'FCGR3A', 'FCGR3B']
    
    # Print marker genes
    print("\nMarker genes for each cluster:")
    for cluster, genes in marker_genes.items():
        print(f"  Cluster {cluster}: {', '.join(genes[:5])}...")
    
    # Run consensus annotation
    print("\nRunning consensus annotation with iterative discussion...")
    result = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species='human',
        tissue='lung',
        models=models,
        api_keys=api_keys,
        consensus_threshold=0.9,  # High consensus threshold to make it harder to reach consensus
        max_discussion_rounds=3,  # Maximum 3 rounds of discussion
        use_cache=True,
        verbose=True
    )
    
    # Print results summary
    print_consensus_summary(result)
    
    # Print detailed discussion logs
    controversial_clusters = result.get("controversial_clusters", [])
    if controversial_clusters:
        print("\n=== Detailed Discussion Logs for Controversial Clusters ===\n")
        for cluster in controversial_clusters:
            if "discussion_logs" in result and cluster in result["discussion_logs"]:
                print(f"\n{'='*80}")
                print(f"Cluster {cluster} Discussion:")
                print(f"{'='*80}")
                
                # Get the complete discussion log
                logs = result["discussion_logs"][cluster]
                
                # Print the complete discussion log
                if isinstance(logs, list):
                    print("\n".join(logs))
                else:
                    print(logs)
                
                # Try to extract and display consensus metrics
                logs_text = logs if isinstance(logs, str) else "\n".join(logs)
                
                print("\n=== Extracted Consensus Metrics ===\n")
                
                # Extract initial metrics
                initial_metrics_found = False
                if "Initial votes" in logs_text:
                    initial_section = logs_text.split("Round 1")[0] if "Round 1" in logs_text else logs_text
                    print("Initial metrics:")
                    for line in initial_section.split("\n"):
                        if "Consensus Proportion (CP):" in line or "Shannon Entropy (H):" in line:
                            print(f"  {line.strip()}")
                            initial_metrics_found = True
                    
                    if not initial_metrics_found:
                        print("  No initial metrics found")
                
                # Extract metrics for each round of discussion
                rounds = logs_text.split("Round ")
                for i in range(1, len(rounds)):
                    round_text = rounds[i]
                    round_metrics_found = False
                    
                    print(f"\nRound {i} metrics:")
                    # Extract proposed cell type
                    for line in round_text.split("\n"):
                        if "Proposed cell type:" in line:
                            print(f"  {line.strip()}")
                            round_metrics_found = True
                        elif "Consensus Proportion (CP):" in line or "Shannon Entropy (H):" in line:
                            print(f"  {line.strip()}")
                            round_metrics_found = True
                    
                    if not round_metrics_found:
                        print("  No metrics found for this round")
    
    # Return results
    return result

if __name__ == "__main__":
    run_schiller_dataset_test()
