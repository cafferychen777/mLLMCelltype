#!/usr/bin/env python3
"""
Test script for mLLMCelltype with Scanpy integration.
Uses API keys from .env file in the mLLMCelltype directory.
"""

import os
import sys

# Set matplotlib to non-interactive mode to avoid plotting blocks
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use Agg backend, which is a non-interactive backend
import matplotlib.pyplot as plt
import scanpy as sc

# Turn off interactive plotting
plt.ioff()

import shutil

from dotenv import load_dotenv

# Add python directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "python")))

# Import mLLMCelltype functions
from mllmcelltype import annotate_clusters, interactive_consensus_annotation, setup_logging

# Load API keys from .env file
# Try to find .env file in various locations
env_path = None

# Try current directory
if os.path.exists(".env"):
    env_path = ".env"

# Try parent directories
if not env_path:
    current_dir = os.path.abspath(os.getcwd())
    for _ in range(3):  # Check up to 3 parent directories
        parent_dir = os.path.dirname(current_dir)
        potential_path = os.path.join(parent_dir, ".env")
        if os.path.exists(potential_path):
            env_path = potential_path
            break
        if parent_dir == current_dir:  # Reached root directory
            break
        current_dir = parent_dir

# Try package directory
if not env_path:
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    potential_path = os.path.join(package_dir, ".env")
    if os.path.exists(potential_path):
        env_path = potential_path

if env_path:
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print("No .env file found. Please set API keys as environment variables.")

# Set up logging
setup_logging()

# Download and load example data (PBMC dataset)
print("Downloading example data...")
adata = sc.datasets.pbmc3k()
print(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")

# Preprocess the data
print("Preprocessing data...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
print(f"Identified {len(adata.obs['leiden'].cat.categories)} clusters")

# Run differential expression analysis to get marker genes
print("Finding marker genes...")
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

# Extract marker genes for each cluster
marker_genes = {}
for i in range(len(adata.obs["leiden"].cat.categories)):
    # Extract top 10 genes for each cluster
    genes = [adata.uns["rank_genes_groups"]["names"][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes
    print(f"Cluster {i} markers: {', '.join(genes[:3])}...")

# Check if API keys are available
api_keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "QWEN_API_KEY": os.getenv("QWEN_API_KEY"),
}

available_apis = [k for k, v in api_keys.items() if v]
print(f"Available API keys: {', '.join(available_apis)}")

if not available_apis:
    print("No API keys found in .env file. Please add your API keys.")
    sys.exit(1)

# Determine which models to use based on available API keys
models = []
if os.getenv("OPENAI_API_KEY"):
    models.append("gpt-4o")
if os.getenv("ANTHROPIC_API_KEY"):
    models.append("claude-3-5-sonnet-latest")
if os.getenv("GEMINI_API_KEY"):
    models.append("gemini-2.0-flash")
if os.getenv("QWEN_API_KEY"):
    models.append("qwen-max")

print(f"Using models: {', '.join(models)}")

if len(models) < 2:
    print("Warning: For consensus annotation, at least 2 models are recommended.")
    # Fall back to single model annotation if only one API key is available
    if len(models) == 1:
        print(f"Performing single model annotation with {models[0]}...")
        provider = (
            "openai"
            if "gpt" in models[0]
            else "anthropic"
            if "claude" in models[0]
            else "gemini"
            if "gemini" in models[0]
            else "qwen"
        )
        annotations = annotate_clusters(
            marker_genes=marker_genes,
            species="human",
            tissue="blood",
            provider=provider,
            model=models[0],
        )

        # Add annotations to AnnData object
        adata.obs["cell_type"] = adata.obs["leiden"].astype(str).map(annotations)

        # Visualize results
        sc.pl.umap(
            adata, color="cell_type", legend_loc="on data", save="_single_model_annotation.png"
        )
        print(f"Results saved as figures/umap_single_model_annotation.png")

        # Print annotations
        print("\nCluster annotations:")
        for cluster, annotation in annotations.items():
            print(f"Cluster {cluster}: {annotation}")

        sys.exit(0)
    else:
        print("No models available. Please add API keys to .env file.")
        sys.exit(1)

# Run consensus annotation with multiple models
print("\nRunning consensus annotation with multiple models...")
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=models,
    consensus_threshold=0.7,  # Adjust threshold for consensus agreement
    max_discussion_rounds=3,  # Maximum rounds of discussion between models
    verbose=True,
)

# Access the final consensus annotations from the dictionary
final_annotations = consensus_results["consensus"]

# Add consensus annotations to AnnData object
adata.obs["consensus_cell_type"] = adata.obs["leiden"].astype(str).map(final_annotations)

# Add consensus proportion and entropy metrics to AnnData object
adata.obs["consensus_proportion"] = (
    adata.obs["leiden"].astype(str).map(consensus_results["consensus_proportion"])
)
adata.obs["entropy"] = adata.obs["leiden"].astype(str).map(consensus_results["entropy"])

# Visualize results
plt.figure(figsize=(12, 10))
sc.pl.umap(
    adata, color="consensus_cell_type", legend_loc="on data", save="_consensus_annotation.png"
)
sc.pl.umap(adata, color="consensus_proportion", save="_consensus_proportion.png")
sc.pl.umap(adata, color="entropy", save="_entropy.png")

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
print("\nTest completed successfully!")
