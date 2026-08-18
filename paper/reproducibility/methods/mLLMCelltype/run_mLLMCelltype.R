# mLLMCelltype Cell Type Annotation Script (R Version)
# Reference: Yang et al. Communications Biology 2026 https://doi.org/10.1038/s42003-026-10420-8
# GitHub: https://github.com/cafferychen777/mLLMCelltype
# Documentation: https://cafferyang.com/mLLMCelltype/

# =============================================================================
# Installation (run once)
# =============================================================================
# install.packages("mLLMCelltype")  # From CRAN
# Or development version:
# devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")

# =============================================================================
# Load required packages
# =============================================================================
library(mLLMCelltype)
library(Seurat)
library(dplyr)

# =============================================================================
# Configuration - Modify these parameters
# =============================================================================

# Input: Path to your Seurat object (.rds file)
SEURAT_OBJECT_PATH <- "path/to/your/seurat_object.rds"  # <-- Modify this

# Output: Path to save annotated results
OUTPUT_PATH <- "path/to/output/mLLMCelltype_results.rds"  # <-- Modify this

# Tissue type (required for context)
# Examples: "human PBMC", "mouse brain", "human lung", etc.
TISSUE_NAME <- "human PBMC"  # <-- Modify this

# API keys for different providers
# At least one is required
API_KEYS <- list(
    openai = "your-openai-key",        # For GPT-5, GPT-4.1, etc.
    anthropic = "your-anthropic-key",  # For Claude models
    gemini = "your-google-key",        # For Gemini models
    qwen = "your-qwen-key"             # For Qwen models
)

# Models to use for consensus annotation
# Supported models:
#   OpenAI: "gpt-5", "gpt-4.1", "o1", "o1-mini"
#   Anthropic: "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022"
#   Google: "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"
#   Alibaba: "qwen-max-2025-01-25"
#   DeepSeek: "deepseek-chat", "deepseek-reasoner"
#   OpenRouter (free): "meta-llama/llama-4-maverick:free", "deepseek/deepseek-r1:free"

MODELS <- c(
    "claude-sonnet-4-5-20250929",
    "gpt-5",
    "gemini-2.5-pro",
    "qwen-max-2025-01-25"
)

# Consensus parameters
CONTROVERSY_THRESHOLD <- 0.7  # Minimum agreement for consensus
ENTROPY_THRESHOLD <- 1.0      # Maximum entropy threshold
MAX_DISCUSSION_ROUNDS <- 3    # Number of discussion rounds

# Cache directory (speeds up repeated runs)
CACHE_DIR <- "./mllmcelltype_cache"

# =============================================================================
# Setup
# =============================================================================
dir.create(CACHE_DIR, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# Load Seurat object
# =============================================================================
cat("Loading Seurat object...\n")
seurat_obj <- readRDS(SEURAT_OBJECT_PATH)

cat("Query data:", ncol(seurat_obj), "cells,", nrow(seurat_obj), "genes\n")

# =============================================================================
# Find marker genes
# =============================================================================
cat("Finding marker genes for each cluster...\n")

markers_file <- file.path(dirname(OUTPUT_PATH), "markers.rds")

if (file.exists(markers_file)) {
    cat("Loading existing markers from:", markers_file, "\n")
    markers <- readRDS(markers_file)
} else {
    cat("Running FindAllMarkers...\n")
    markers <- FindAllMarkers(
        seurat_obj,
        only.pos = TRUE,
        min.pct = 0.25,
        logfc.threshold = 0.25
    )
    saveRDS(markers, markers_file)
    cat("Markers saved to:", markers_file, "\n")
}

cat("Found markers for", length(unique(markers$cluster)), "clusters\n")

# =============================================================================
# Run mLLMCelltype consensus annotation
# =============================================================================
cat("Running mLLMCelltype consensus annotation...\n")
cat("Models:", paste(MODELS, collapse = ", "), "\n")
cat("Tissue:", TISSUE_NAME, "\n")

consensus_results <- interactive_consensus_annotation(
    input = markers,
    tissue_name = TISSUE_NAME,
    models = MODELS,
    api_keys = API_KEYS,
    top_gene_count = 10,
    controversy_threshold = CONTROVERSY_THRESHOLD,
    entropy_threshold = ENTROPY_THRESHOLD,
    max_discussion_rounds = MAX_DISCUSSION_ROUNDS,
    cache_dir = CACHE_DIR
)

# =============================================================================
# Add annotations to Seurat object
# =============================================================================
cat("Adding annotations to Seurat object...\n")

# Get cell type annotations
cluster_to_celltype <- consensus_results$final_annotations

# Map to all cells
cell_types <- as.character(Idents(seurat_obj))
for (cluster_id in names(cluster_to_celltype)) {
    cell_types[cell_types == cluster_id] <- cluster_to_celltype[[cluster_id]]
}
seurat_obj$mLLMCelltype <- cell_types

# Add uncertainty metrics
consensus_details <- consensus_results$initial_results$consensus_results
uncertainty_df <- data.frame(
    cluster_id = names(consensus_details),
    consensus_proportion = sapply(consensus_details, function(x) x$consensus_proportion),
    entropy = sapply(consensus_details, function(x) x$entropy)
)

current_clusters <- as.character(seurat_obj$seurat_clusters)
seurat_obj$consensus_proportion <- uncertainty_df$consensus_proportion[
    match(current_clusters, uncertainty_df$cluster_id)
]
seurat_obj$entropy <- uncertainty_df$entropy[
    match(current_clusters, uncertainty_df$cluster_id)
]

# =============================================================================
# Results summary
# =============================================================================
cat("\n=== Annotation Results ===\n")
print(table(seurat_obj$mLLMCelltype))

cat("\n=== Cluster to Cell Type Mapping ===\n")
for (cluster in names(cluster_to_celltype)) {
    cat(sprintf("Cluster %s: %s\n", cluster, cluster_to_celltype[[cluster]]))
}

cat("\n=== Uncertainty Metrics ===\n")
print(uncertainty_df)

# =============================================================================
# Save results
# =============================================================================
cat("\nSaving results to:", OUTPUT_PATH, "\n")
saveRDS(seurat_obj, OUTPUT_PATH)

# Save consensus results separately
consensus_path <- sub("\\.rds$", "_consensus.rds", OUTPUT_PATH)
saveRDS(consensus_results, consensus_path)
cat("Consensus results saved to:", consensus_path, "\n")

cat("\nDone!\n")
