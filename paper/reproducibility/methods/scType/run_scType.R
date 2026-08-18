# scType Cell Type Annotation Script
# Reference: Ianevski et al. (2022) https://doi.org/10.1038/s41467-022-28803-w
# GitHub: https://github.com/IanevskiAleksandr/sc-type

# =============================================================================
# Load required packages
# =============================================================================
library(Seurat)
library(HGNChelper)
library(openxlsx)

# Load scType functions from GitHub
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

# =============================================================================
# Configuration - Modify these parameters
# =============================================================================

# Input: Path to your Seurat object (.rds file)
SEURAT_OBJECT_PATH <- "path/to/your/seurat_object.rds"  # <-- Modify this

# Output: Path to save annotated results
OUTPUT_PATH <- "path/to/output/scType_results.rds"  # <-- Modify this

# Tissue type selection
# Available options:
#   - "Immune system"
#   - "Pancreas"
#   - "Liver"
#   - "Eye"
#   - "Kidney"
#   - "Brain"
#   - "Lung"
#   - "Adrenal"
#   - "Heart"
#   - "Intestine"
#   - "Muscle"
#   - "Placenta"
#   - "Spleen"
#   - "Stomach"
#   - "Thymus"

TISSUE_TYPE <- "Immune system"  # <-- Modify this

# Marker database (use default or provide custom path)
MARKER_DB <- "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

# Clustering resolution in Seurat object (column name)
CLUSTER_COLUMN <- "seurat_clusters"  # <-- Modify if different

# =============================================================================
# Load and prepare query data
# =============================================================================
cat("Loading Seurat object...\n")
seurat_obj <- readRDS(SEURAT_OBJECT_PATH)

cat("Query data:", ncol(seurat_obj), "cells\n")

# Ensure data is scaled
if (!"scale.data" %in% names(seurat_obj@assays$RNA@layers) &&
    length(seurat_obj@assays$RNA@scale.data) == 0) {
    cat("Scaling data...\n")
    seurat_obj <- ScaleData(seurat_obj)
}

# =============================================================================
# Prepare gene sets
# =============================================================================
cat("Loading marker gene sets for tissue:", TISSUE_TYPE, "\n")
gs_list <- gene_sets_prepare(MARKER_DB, TISSUE_TYPE)

cat("Loaded", length(gs_list$gs_positive), "cell types\n")

# =============================================================================
# Run scType scoring
# =============================================================================
cat("Running scType scoring...\n")

# Get scaled data matrix
scRNAseqData_scaled <- as.matrix(GetAssayData(seurat_obj, layer = "scale.data"))

# Calculate scType scores
es.max <- sctype_score(
    scRNAseqData = scRNAseqData_scaled,
    scaled = TRUE,
    gs = gs_list$gs_positive,
    gs2 = gs_list$gs_negative
)

# =============================================================================
# Assign cell types to clusters
# =============================================================================
cat("Assigning cell types to clusters...\n")

# Get cluster assignments
clusters <- seurat_obj@meta.data[[CLUSTER_COLUMN]]

# Calculate scores per cluster
cL_results <- do.call("rbind", lapply(unique(clusters), function(cl) {
    es.max.cl <- sort(
        rowSums(es.max[, rownames(seurat_obj@meta.data[clusters == cl, ]), drop = FALSE]),
        decreasing = TRUE
    )
    head(data.frame(
        cluster = cl,
        type = names(es.max.cl),
        scores = es.max.cl,
        ncells = sum(clusters == cl)
    ), 10)
}))

# Select top cell type for each cluster
sctype_scores <- cL_results %>%
    dplyr::group_by(cluster) %>%
    dplyr::top_n(n = 1, wt = scores)

# Set low-confidence assignments to "Unknown"
sctype_scores$type[sctype_scores$scores < sctype_scores$ncells / 4] <- "Unknown"

# Add cluster-level annotations to Seurat object
seurat_obj@meta.data$sctype_classification <- NA
for (i in seq_len(nrow(sctype_scores))) {
    cl <- sctype_scores$cluster[i]
    seurat_obj@meta.data$sctype_classification[clusters == cl] <- sctype_scores$type[i]
}

# =============================================================================
# Results summary
# =============================================================================
cat("\n=== Annotation Results ===\n")
print(table(seurat_obj$sctype_classification))

cat("\n=== Per-cluster Results ===\n")
print(sctype_scores[, c("cluster", "type", "scores", "ncells")])

# =============================================================================
# Save results
# =============================================================================
cat("\nSaving results to:", OUTPUT_PATH, "\n")
saveRDS(seurat_obj, OUTPUT_PATH)

# Also save detailed scoring results
scores_path <- sub("\\.rds$", "_scores.rds", OUTPUT_PATH)
saveRDS(list(
    es.max = es.max,
    cL_results = cL_results,
    sctype_scores = sctype_scores
), scores_path)
cat("Detailed scores saved to:", scores_path, "\n")

cat("\nDone!\n")
