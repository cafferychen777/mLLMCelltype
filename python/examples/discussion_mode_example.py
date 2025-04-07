#!/usr/bin/env python

"""
Test the discussion mode functionality of LLMCelltype.
Force trigger discussion mode by setting a high consensus threshold.
"""

import os
import sys
import numpy as np
import pandas as pd

# Set matplotlib to non-interactive mode to avoid plotting blocks
import matplotlib
matplotlib.use('Agg')  # Use Agg backend, which is a non-interactive backend
import matplotlib.pyplot as plt
import scanpy as sc

# Turn off interactive plotting
plt.ioff()

from dotenv import load_dotenv
import shutil

# Add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from mllmcelltype.consensus import interactive_consensus_annotation

# Load API keys from .env file
# Try to find .env file in various locations
env_path = None

# Try current directory
if os.path.exists('.env'):
    env_path = '.env'

# Try parent directories
if not env_path:
    current_dir = os.path.abspath(os.getcwd())
    for _ in range(3):  # Check up to 3 parent directories
        parent_dir = os.path.dirname(current_dir)
        potential_path = os.path.join(parent_dir, '.env')
        if os.path.exists(potential_path):
            env_path = potential_path
            break
        if parent_dir == current_dir:  # Reached root directory
            break
        current_dir = parent_dir

# Try package directory
if not env_path:
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    potential_path = os.path.join(package_dir, '.env')
    if os.path.exists(potential_path):
        env_path = potential_path

if env_path:
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print("No .env file found. Please set API keys as environment variables.")

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

# Download example data
def download_example_data():
    """Download example dataset"""
    print("Downloading example data...")
    # Use scanpy's built-in PBMC dataset
    adata = sc.datasets.pbmc3k()
    return adata

# Preprocess data
def preprocess_data(adata):
    """Preprocess single-cell data"""
    print("Preprocessing data...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Calculate quality control metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter cells
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # Dimensionality reduction and clustering
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    
    print(f"Identified {len(adata.obs['leiden'].unique())} clusters")
    return adata

# Find marker genes for each cluster
def find_marker_genes(adata):
    """Find marker genes for each cluster"""
    print("Finding marker genes...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    # Extract top 10 marker genes for each cluster
    marker_genes = {}
    for i in range(len(adata.obs['leiden'].unique())):
        cluster_id = str(i)
        markers = [gene for gene in adata.uns['rank_genes_groups']['names'][cluster_id][:20]]
        marker_genes[cluster_id] = markers
        print(f"Cluster {cluster_id} markers: {', '.join(markers[:3])}...")
    
    return marker_genes

def main():
    # Download and preprocess data
    adata = download_example_data()
    adata = preprocess_data(adata)
    marker_genes = find_marker_genes(adata)
    
    # Check available API keys
    api_keys = {}
    for provider in ['openai', 'anthropic', 'gemini', 'qwen']:
        env_var = f"{provider.upper()}_API_KEY"
        if os.environ.get(env_var):
            api_keys[provider] = os.environ.get(env_var)
    
    print(f"Available API keys: {', '.join(api_keys.keys())}")
    
    # Select models to use
    models = []
    if 'openai' in api_keys:
        models.append('gpt-4o')
    if 'anthropic' in api_keys:
        models.append('claude-3-5-sonnet-latest')
    if 'gemini' in api_keys:
        models.append('gemini-1.5-pro')
    if 'qwen' in api_keys:
        models.append('qwen-max')
    
    print(f"Using models: {', '.join(models)}")
    
    # Set consensus threshold to 1.0, only complete agreement is considered consensus
    consensus_threshold = 1.0  # Set to the highest threshold
    
    print("\nRunning consensus annotation with discussion mode (high threshold)...")
    # Run consensus annotation
    consensus_results = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human",
        models=models,
        api_keys=api_keys,
        tissue="blood",
        consensus_threshold=consensus_threshold,  # Use high threshold
        max_discussion_rounds=5,  # Maximum 5 rounds of discussion
        use_cache=True,
        verbose=True
    )
    
    # Extract final annotations
    final_annotations = consensus_results["consensus"]
    
    # Add consensus annotations to AnnData object
    adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)
    
    # Add consensus proportion and entropy metrics to AnnData object
    adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])
    adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])
    
    # Visualize results
    plt.figure(figsize=(12, 10))
    sc.pl.umap(adata, color='consensus_cell_type', legend_loc='on data', save="_consensus_annotation.png")
    sc.pl.umap(adata, color='consensus_proportion', save="_consensus_proportion.png")
    sc.pl.umap(adata, color='entropy', save="_entropy.png")
    
    print("\nResults saved as:")
    print("- figures/umap_consensus_annotation.png")
    print("- figures/umap_consensus_proportion.png")
    print("- figures/umap_entropy.png")
    
    # Print consensus annotations and uncertainty metrics
    print("\nConsensus annotations with uncertainty metrics:")
    for cluster in sorted(final_annotations.keys(), key=int):
        cp = consensus_results["consensus_proportion"][cluster]
        entropy = consensus_results["entropy"][cluster]
        print(f"Cluster {cluster}: {final_annotations[cluster]} (CP: {cp:.2f}, Entropy: {entropy:.2f})")
    
    # Print discussion logs
    print("\nDiscussion logs for controversial clusters:")
    discussion_logs = consensus_results.get("discussion_logs", {})
    if discussion_logs:
        for cluster, logs in discussion_logs.items():
            print(f"\nCluster {cluster} discussion:")
            for round_num, log in enumerate(logs):
                print(f"  Round {round_num+1}:")
                print(f"  {log[:100]}...")  # Only print the first 100 characters
    else:
        print("No discussion logs found.")
    
    # Save results
    result_file = "discussion_results.txt"
    with open(result_file, "w") as f:
        f.write("Cluster\tCell Type\tConsensus Proportion\tEntropy\n")
        for cluster in sorted(final_annotations.keys(), key=int):
            cp = consensus_results["consensus_proportion"][cluster]
            entropy = consensus_results["entropy"][cluster]
            f.write(f"{cluster}\t{final_annotations[cluster]}\t{cp:.2f}\t{entropy:.2f}\n")
        
        f.write("\n\nDiscussion Logs:\n")
        for cluster, logs in discussion_logs.items():
            f.write(f"\nCluster {cluster} discussion:\n")
            for round_num, log in enumerate(logs):
                f.write(f"Round {round_num+1}:\n{log}\n")
    
    print(f"\nDetailed results saved to {result_file}")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
