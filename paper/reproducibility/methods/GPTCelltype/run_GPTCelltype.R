# GPTCelltype Cell Type Annotation Script
# Reference: Hou & Ji, Nature Methods 2024 https://doi.org/10.1038/s41592-024-02235-4
# GitHub: https://github.com/Winnie09/GPTCelltype

# =============================================================================
# Installation (run once)
# =============================================================================
# install.packages("openai")
# remotes::install_github("Winnie09/GPTCelltype")

# =============================================================================
# Load required packages
# =============================================================================
library(Seurat)
library(GPTCelltype)
library(openai)

# =============================================================================
# Configuration - Modify these parameters
# =============================================================================

# IMPORTANT: Set your OpenAI API key
# Option 1: Set in R (not recommended for shared code)
# Sys.setenv(OPENAI_API_KEY = "your_openai_API_key")
# Option 2: Set in .Renviron file (recommended)
# OPENAI_API_KEY=your_openai_API_key

# Input: Path to your Seurat object (.rds file)
SEURAT_OBJECT_PATH <- "path/to/your/seurat_object.rds"  # <-- Modify this

# Output: Path to save annotated results
OUTPUT_PATH <- "path/to/output/GPTCelltype_results.rds"  # <-- Modify this

# Model selection
# Options: "gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
MODEL <- "gpt-4"  # <-- Modify this

# Tissue type (optional, improves accuracy)
# Examples: "human PBMC", "mouse brain", "human lung", etc.
# Set to NULL if unknown
TISSUE_NAME <- "human PBMC"  # <-- Modify this

# =============================================================================
# Verify API key is set
# =============================================================================
if (Sys.getenv("OPENAI_API_KEY") == "") {
    stop("OPENAI_API_KEY not set. Please set it using Sys.setenv() or .Renviron file.")
}

# =============================================================================
# Load Seurat object
# =============================================================================
cat("Loading Seurat object...\n")
seurat_obj <- readRDS(SEURAT_OBJECT_PATH)

cat("Query data:", ncol(seurat_obj), "cells,", nrow(seurat_obj), "genes\n")

# =============================================================================
# Find marker genes (if not already done)
# =============================================================================
cat("Finding marker genes for each cluster...\n")

# Check if markers already exist
markers_file <- sub("\\.rds$", "_markers.rds", OUTPUT_PATH)

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
# Run GPTCelltype annotation
# =============================================================================
cat("Running GPTCelltype with model:", MODEL, "\n")

if (!is.null(TISSUE_NAME)) {
    cat("Using tissue context:", TISSUE_NAME, "\n")
    res <- gptcelltype(markers, model = MODEL, tissuename = TISSUE_NAME)
} else {
    res <- gptcelltype(markers, model = MODEL)
}

# =============================================================================
# Add annotations to Seurat object
# =============================================================================
cat("Adding annotations to Seurat object...\n")

# Map cluster IDs to cell type names
seurat_obj$GPTCelltype <- as.factor(res[as.character(Idents(seurat_obj))])

# =============================================================================
# Results summary
# =============================================================================
cat("\n=== Annotation Results ===\n")
print(table(seurat_obj$GPTCelltype))

cat("\n=== Cluster to Cell Type Mapping ===\n")
print(res)

# =============================================================================
# Save results
# =============================================================================
cat("\nSaving results to:", OUTPUT_PATH, "\n")
saveRDS(seurat_obj, OUTPUT_PATH)

# Also save the raw GPTCelltype results
results_path <- sub("\\.rds$", "_gptcelltype_raw.rds", OUTPUT_PATH)
saveRDS(res, results_path)
cat("Raw GPTCelltype results saved to:", results_path, "\n")

cat("\nDone!\n")
