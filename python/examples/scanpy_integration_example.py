#!/usr/bin/env python3
"""
Example of mLLMCelltype integration with Scanpy for single-cell RNA-seq analysis.
This example demonstrates how to:
1. Preprocess single-cell data with Scanpy
2. Extract marker genes from clusters
3. Perform cell type annotation using multiple LLMs
4. Visualize and interpret the results

Requirements:
- scanpy
- matplotlib
- numpy
- pandas
- dotenv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from dotenv import load_dotenv

# Set matplotlib to non-interactive mode
plt.ioff()
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import mLLMCelltype functions
from mllmcelltype import annotate_clusters, setup_logging, interactive_consensus_annotation

def run_scanpy_example():
    """Run the complete Scanpy integration example"""
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Load API keys from .env file
    load_dotenv()
    print("Checking for available API keys...")
    
    # Check which API keys are available
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "QWEN_API_KEY": os.getenv("QWEN_API_KEY")
    }
    
    available_apis = [k.split('_')[0].lower() for k, v in api_keys.items() if v]
    print(f"Available API providers: {', '.join(available_apis)}")
    
    if not available_apis:
        print("No API keys found. Please add your API keys to .env file or environment variables.")
        return
    
    # Determine which models to use based on available API keys
    models = []
    if os.getenv("OPENAI_API_KEY"):
        models.append("gpt-4o")
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append("claude-3-5-sonnet-latest")
    if os.getenv("GEMINI_API_KEY"):
        models.append("gemini-1.5-pro")
    if os.getenv("QWEN_API_KEY"):
        models.append("qwen-max")
    
    print(f"Using models: {', '.join(models)}")
    
    # Download and load example data (PBMC dataset)
    print("\nDownloading example data (PBMC)...")
    adata = sc.datasets.pbmc3k()
    print(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
    
    # Preprocess the data
    print("\nPreprocessing data...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    print(f"Identified {len(adata.obs['leiden'].cat.categories)} clusters")
    
    # Run differential expression analysis to get marker genes
    print("\nFinding marker genes...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    # Extract marker genes for each cluster
    marker_genes = {}
    for i in range(len(adata.obs['leiden'].cat.categories)):
        # Extract top 10 genes for each cluster
        genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
        marker_genes[str(i)] = genes
        print(f"Cluster {i} markers: {', '.join(genes[:3])}...")
    
    # Create output directory for figures
    os.makedirs("figures", exist_ok=True)
    
    # Handle different scenarios based on available models
    if len(models) < 2:
        run_single_model_annotation(adata, marker_genes, models, available_apis)
    else:
        run_consensus_annotation(adata, marker_genes, models)

def run_single_model_annotation(adata, marker_genes, models, available_apis):
    """Run annotation with a single model"""
    print("\nRunning single model annotation...")
    
    if len(models) == 0:
        print("No models available. Please add API keys.")
        return
    
    model = models[0]
    provider = [api for api in available_apis if api in model.lower()][0]
    if not provider:
        provider = available_apis[0]
    
    print(f"Using model {model} from provider {provider}")
    
    # Perform annotation
    annotations = annotate_clusters(
        marker_genes=marker_genes,
        species="human",
        tissue="blood",
        provider=provider,
        model=model
    )
    
    # Add annotations to AnnData object
    adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)
    
    # Visualize results
    plt.figure(figsize=(12, 10))
    sc.pl.umap(adata, color='cell_type', legend_loc='on data', save="_single_model_annotation.png")
    
    print("\nResults saved as figures/umap_single_model_annotation.png")
    
    # Print annotations
    print("\nCluster annotations:")
    for cluster, annotation in annotations.items():
        print(f"Cluster {cluster}: {annotation}")

def run_consensus_annotation(adata, marker_genes, models):
    """Run consensus annotation with multiple models"""
    print("\nRunning consensus annotation with multiple models...")
    
    consensus_results = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human",
        tissue="blood",
        models=models,
        consensus_threshold=0.7,  # Adjust threshold for consensus agreement
        max_discussion_rounds=3,  # Maximum rounds of discussion between models
        verbose=True
    )
    
    # Access the final consensus annotations from the dictionary
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
    
    # Print consensus annotations with uncertainty metrics
    print("\nConsensus annotations with uncertainty metrics:")
    for cluster in sorted(final_annotations.keys(), key=int):
        cp = consensus_results["consensus_proportion"][cluster]
        entropy = consensus_results["entropy"][cluster]
        print(f"Cluster {cluster}: {final_annotations[cluster]} (CP: {cp:.2f}, Entropy: {entropy:.2f})")
    
    # Save results
    result_file = "consensus_results.txt"
    with open(result_file, "w") as f:
        f.write("Cluster\tCell Type\tConsensus Proportion\tEntropy\n")
        for cluster in sorted(final_annotations.keys(), key=int):
            cp = consensus_results["consensus_proportion"][cluster]
            entropy = consensus_results["entropy"][cluster]
            f.write(f"{cluster}\t{final_annotations[cluster]}\t{cp:.2f}\t{entropy:.2f}\n")
    
    print(f"\nDetailed results saved to {result_file}")

if __name__ == "__main__":
    print("mLLMCelltype Scanpy Integration Example")
    print("=======================================")
    run_scanpy_example()
    print("\nExample completed successfully!")
