# SingleR Cell Type Annotation Script
# Reference: Aran et al. (2019) https://doi.org/10.1038/s41590-018-0276-y
# Documentation: https://bioconductor.org/packages/release/bioc/html/SingleR.html

# =============================================================================
# Installation (run once)
# =============================================================================
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# BiocManager::install(c("SingleR", "celldex", "SingleCellExperiment"))

# =============================================================================
# Load required packages
# =============================================================================
library(Seurat)
library(SingleR)
library(celldex)
library(SingleCellExperiment)

# =============================================================================
# Configuration - Modify these parameters
# =============================================================================

# Input: Path to your Seurat object (.rds file)
SEURAT_OBJECT_PATH <- "path/to/your/seurat_object.rds"  # <-- Modify this

# Output: Path to save annotated results
OUTPUT_PATH <- "path/to/output/singleR_results.rds"  # <-- Modify this

# Reference dataset selection
# Options for HUMAN data:
#   - "HumanPrimaryCellAtlas" : 713 samples, 37 main types (broad coverage)
#   - "BlueprintEncode"       : 259 samples, 24 main types (immune + stroma)
#   - "MonacoImmune"          : Immune cell focused
#   - "DatabaseImmuneCellExpression" : Immune cells
#   - "NovershternHematopoietic" : Hematopoietic cells
#
# Options for MOUSE data:
#   - "ImmGen"        : Immunological Genome Project (immune cells)
#   - "MouseRNAseq"   : Mouse brain cell types

REFERENCE_TYPE <- "HumanPrimaryCellAtlas"  # <-- Modify this

# Label granularity: "main" (broad) or "fine" (detailed)
LABEL_TYPE <- "main"  # <-- Modify this

# =============================================================================
# Load reference dataset
# =============================================================================
cat("Loading reference dataset:", REFERENCE_TYPE, "\n")

ref_data <- switch(REFERENCE_TYPE,
    "HumanPrimaryCellAtlas" = celldex::HumanPrimaryCellAtlasData(),
    "BlueprintEncode" = celldex::BlueprintEncodeData(),
    "MonacoImmune" = celldex::MonacoImmuneData(),
    "DatabaseImmuneCellExpression" = celldex::DatabaseImmuneCellExpressionData(),
    "NovershternHematopoietic" = celldex::NovershternHematopoieticData(),
    "ImmGen" = celldex::ImmGenData(),
    "MouseRNAseq" = celldex::MouseRNAseqData(),
    stop("Unknown reference type: ", REFERENCE_TYPE)
)

# Get labels based on granularity
labels <- if (LABEL_TYPE == "main") {
    ref_data$label.main
} else {
    ref_data$label.fine
}

cat("Reference loaded:", ncol(ref_data), "samples,",
    length(unique(labels)), "cell types\n")

# =============================================================================
# Load and prepare query data
# =============================================================================
cat("Loading Seurat object...\n")
seurat_obj <- readRDS(SEURAT_OBJECT_PATH)

# Convert to SingleCellExperiment
sce <- as.SingleCellExperiment(seurat_obj)

cat("Query data:", ncol(sce), "cells\n")

# =============================================================================
# Run SingleR annotation
# =============================================================================
cat("Running SingleR annotation...\n")

predictions <- SingleR(
    test = sce,
    ref = ref_data,
    labels = labels,
    assay.type.test = "logcounts"  # Use log-normalized counts
)

# =============================================================================
# Add predictions to Seurat object
# =============================================================================
seurat_obj$SingleR_labels <- predictions$labels
seurat_obj$SingleR_pruned_labels <- predictions$pruned.labels  # NA for low-quality

# Summary of predictions
cat("\n=== Annotation Results ===\n")
print(table(predictions$labels))

# Check annotation quality (pruned labels indicate low confidence)
n_pruned <- sum(is.na(predictions$pruned.labels))
cat("\nLow confidence annotations (pruned):", n_pruned,
    "(", round(100 * n_pruned / ncol(sce), 1), "%)\n")

# =============================================================================
# Save results
# =============================================================================
cat("\nSaving results to:", OUTPUT_PATH, "\n")
saveRDS(seurat_obj, OUTPUT_PATH)

# Also save raw SingleR results for detailed analysis
singleR_results_path <- sub("\\.rds$", "_singleR_raw.rds", OUTPUT_PATH)
saveRDS(predictions, singleR_results_path)
cat("Raw SingleR results saved to:", singleR_results_path, "\n")

cat("\nDone!\n")
