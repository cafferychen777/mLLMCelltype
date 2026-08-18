# mLLMCelltype Cell Type Annotation Script (Python Version)
# Reference: Yang et al. Communications Biology 2026 https://doi.org/10.1038/s42003-026-10420-8
# GitHub: https://github.com/cafferychen777/mLLMCelltype
# PyPI: https://pypi.org/project/mllmcelltype/

# =============================================================================
# Installation
# =============================================================================
# pip install mllmcelltype
# Or with specific providers:
# pip install "mllmcelltype[openai]"
# pip install "mllmcelltype[anthropic]"
# pip install "mllmcelltype[all]"

# =============================================================================
# Imports
# =============================================================================
import os
import scanpy as sc
from mllmcelltype import interactive_consensus_annotation, annotate_clusters

# =============================================================================
# Configuration - Modify these parameters
# =============================================================================

# Input: Path to your AnnData object (.h5ad file)
ADATA_PATH = "path/to/your/adata.h5ad"  # <-- Modify this

# Output: Path to save annotated results
OUTPUT_PATH = "path/to/output/mLLMCelltype_results.h5ad"  # <-- Modify this

# Species and tissue context
SPECIES = "human"  # <-- Modify this ("human" or "mouse")
TISSUE = "peripheral blood"  # <-- Modify this (e.g., "brain", "lung", "PBMC")

# API keys - Set at least one
# Option 1: Environment variables (recommended)
# export OPENAI_API_KEY="your-key"
# export ANTHROPIC_API_KEY="your-key"
# export GEMINI_API_KEY="your-key"

# Option 2: Set directly in script
os.environ["OPENAI_API_KEY"] = "your-openai-key"  # <-- Modify this
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"  # <-- Modify this
os.environ["GEMINI_API_KEY"] = "your-google-key"  # <-- Modify this
os.environ["QWEN_API_KEY"] = "your-qwen-key"  # <-- Modify this

# Models to use for consensus annotation
# Supported models:
#   OpenAI: "gpt-5", "gpt-4.1", "o1"
#   Anthropic: "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022"
#   Google: "gemini-2.5-pro", "gemini-2.5-flash"
#   Alibaba: "qwen-max-2025-01-25"
#   DeepSeek: "deepseek-chat", "deepseek-reasoner"
#   OpenRouter (free): Use dict format, e.g.:
#     {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"}

MODELS = [
    "gpt-5",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-pro",
    "qwen-max-2025-01-25"
]

# Consensus parameters
CONSENSUS_THRESHOLD = 0.7  # Minimum agreement for consensus
MAX_DISCUSSION_ROUNDS = 3  # Number of discussion rounds

# =============================================================================
# Load and preprocess data
# =============================================================================
print(f"Loading AnnData from: {ADATA_PATH}")
adata = sc.read_h5ad(ADATA_PATH)

print(f"Query data: {adata.n_obs} cells, {adata.n_vars} genes")

# Preprocessing if needed
if "log1p" not in adata.uns:
    print("Preprocessing: normalizing and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# Clustering if needed
if "leiden" not in adata.obs.columns:
    print("Computing leiden clustering...")
    if "X_pca" not in adata.obsm:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.pca(adata, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.leiden(adata, resolution=0.8)
    print(f"Found {len(adata.obs['leiden'].cat.categories)} clusters")

# =============================================================================
# Extract marker genes
# =============================================================================
print("Finding marker genes for each cluster...")

sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

marker_genes = {}
for cluster in adata.obs["leiden"].cat.categories:
    # Get top 10 marker genes for each cluster
    genes = adata.uns["rank_genes_groups"]["names"][cluster][:10].tolist()
    marker_genes[str(cluster)] = genes

print(f"Extracted markers for {len(marker_genes)} clusters")

# =============================================================================
# Run mLLMCelltype consensus annotation
# =============================================================================
print("Running mLLMCelltype consensus annotation...")
print(f"Models: {MODELS}")
print(f"Species: {SPECIES}, Tissue: {TISSUE}")

consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species=SPECIES,
    tissue=TISSUE,
    models=MODELS,
    consensus_threshold=CONSENSUS_THRESHOLD,
    max_discussion_rounds=MAX_DISCUSSION_ROUNDS,
    verbose=True
)

# =============================================================================
# Add annotations to AnnData
# =============================================================================
print("Adding annotations to AnnData...")

# Cell type annotations
adata.obs["mLLMCelltype"] = adata.obs["leiden"].astype(str).map(
    consensus_results["consensus"]
)

# Uncertainty metrics
adata.obs["consensus_proportion"] = adata.obs["leiden"].astype(str).map(
    consensus_results["consensus_proportion"]
)
adata.obs["entropy"] = adata.obs["leiden"].astype(str).map(
    consensus_results["entropy"]
)

# =============================================================================
# Results summary
# =============================================================================
print("\n=== Annotation Results ===")
print(adata.obs["mLLMCelltype"].value_counts())

print("\n=== Cluster to Cell Type Mapping ===")
for cluster, cell_type in consensus_results["consensus"].items():
    print(f"Cluster {cluster}: {cell_type}")

print("\n=== Uncertainty Metrics ===")
for cluster in marker_genes.keys():
    cp = consensus_results["consensus_proportion"].get(cluster, "N/A")
    entropy = consensus_results["entropy"].get(cluster, "N/A")
    print(f"Cluster {cluster}: CP={cp:.2f}, Entropy={entropy:.2f}")

# =============================================================================
# Visualization (optional)
# =============================================================================
# Compute UMAP if needed
if "X_umap" not in adata.obsm:
    print("\nComputing UMAP...")
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)

# Save UMAP plots
print("\nGenerating visualizations...")
sc.pl.umap(
    adata,
    color="mLLMCelltype",
    legend_loc="on data",
    title="mLLMCelltype Consensus Annotations",
    save="_mLLMCelltype.png"
)

sc.pl.umap(
    adata,
    color=["consensus_proportion", "entropy"],
    title=["Consensus Proportion", "Shannon Entropy"],
    save="_uncertainty.png"
)

# =============================================================================
# Save results
# =============================================================================
print(f"\nSaving annotated data to: {OUTPUT_PATH}")
adata.write(OUTPUT_PATH)

print("\nDone!")
