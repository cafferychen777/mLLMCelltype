#!/usr/bin/env Rscript
# Run SingleR & scType on a single dataset on arseven.
# Usage: Rscript run_singler_sctype_single.R <dataset_name> <h5ad_path> <label_col> <tissue> <output_dir>
#
# Example:
#   Rscript run_singler_sctype_single.R "Bladder(TS)" \
#     /scratch/user/$USER/singler_sctype_batch/data/TS_Bladder_filtered.h5ad \
#     cell_ontology_class "Immune system" \
#     /scratch/user/$USER/singler_sctype_batch/results

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript run_singler_sctype_single.R <name> <h5ad> <label> <tissue> <outdir>")
}

ds_name   <- args[1]
h5ad_path <- args[2]
label_col <- args[3]
tissue    <- args[4]
output_dir <- args[5]

# Ensure user library is on path
.libPaths(c("~/R/4.4", .libPaths()))

suppressPackageStartupMessages({
  library(SingleR)
  library(celldex)
  library(SingleCellExperiment)
  library(scuttle)
  library(scran)
  library(Matrix)
  library(HGNChelper)
  library(openxlsx)
})

# scType helper functions
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load reference
cat("Loading SingleR reference (HumanPrimaryCellAtlasData)...\n")
ref_hpca <- HumanPrimaryCellAtlasData()

# scType DB
DB_URL <- "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

cat(sprintf("\n===== Processing: %s =====\n", ds_name))
cat(sprintf("  File: %s\n", h5ad_path))

if (!file.exists(h5ad_path)) {
  stop(sprintf("File not found: %s", h5ad_path))
}

# Load h5ad using reticulate directly (avoid anndata R package dependency issues)
library(reticulate)
ad <- import("anndata")
adata <- ad$read_h5ad(h5ad_path)
cat(sprintf("  Loaded: %d cells x %d genes\n", adata$n_obs, adata$n_vars))

# Get expression matrix
raw_mat <- adata$X
if (!inherits(raw_mat, "dgCMatrix")) {
  raw_mat <- as(raw_mat, "CsparseMatrix")
}
expr_matrix <- as(t(raw_mat), "CsparseMatrix")  # genes x cells
rownames(expr_matrix) <- adata$var_names$to_list()
colnames(expr_matrix) <- adata$obs_names$to_list()

# Get reference labels
ref_labels <- as.character(adata$obs[[label_col]]$to_list())
clusters <- as.factor(ref_labels)
cat(sprintf("  Unique clusters: %d\n", length(unique(clusters))))

# Create SCE
sce <- SingleCellExperiment(assays = list(counts = expr_matrix))
sce <- computeSumFactors(sce)
sce <- logNormCounts(sce)

# --- SingleR ---
cat("  Running SingleR...\n")
t0 <- Sys.time()
pred <- SingleR(
  test = sce,
  ref = ref_hpca,
  labels = ref_hpca$label.main,
  de.method = "wilcox"
)
cell_labels <- pred$labels
singler_preds <- tapply(cell_labels, clusters, function(x) {
  tab <- table(x)
  names(tab)[which.max(tab)]
})
t1 <- Sys.time()
cat(sprintf("  SingleR done in %.1f min\n", as.numeric(difftime(t1, t0, units = "mins"))))

# --- scType ---
cat("  Running scType...\n")
t0 <- Sys.time()
sctype_preds <- tryCatch({
  gs_list <- gene_sets_prepare(DB_URL, tissue)
  log_counts <- as.matrix(logcounts(sce))
  scaled_matrix <- t(scale(t(log_counts)))
  scaled_matrix[is.nan(scaled_matrix)] <- 0

  es.max <- sctype_score(
    scRNAseqData = scaled_matrix,
    scaled = TRUE,
    gs = gs_list$gs_positive,
    gs2 = gs_list$gs_negative
  )

  unique_clusters <- sort(unique(clusters))
  cl_labels <- character(length(unique_clusters))
  names(cl_labels) <- unique_clusters

  for (cl in unique_clusters) {
    cell_idx <- which(clusters == cl)
    if (length(cell_idx) == 0) next
    scores <- rowMeans(es.max[, cell_idx, drop = FALSE])
    best <- names(which.max(scores))
    n_cells <- length(cell_idx)
    if (max(scores) < n_cells / 4) {
      cl_labels[cl] <- "Unknown"
    } else {
      cl_labels[cl] <- best
    }
  }
  cl_labels
}, error = function(e) {
  cat(sprintf("  scType error: %s\n", e$message))
  NULL
})
t1 <- Sys.time()
cat(sprintf("  scType done in %.1f min\n", as.numeric(difftime(t1, t0, units = "mins"))))

# Save predictions
ref_cluster_labels <- names(singler_preds)
pred_df <- data.frame(
  cluster = ref_cluster_labels,
  reference = ref_cluster_labels,
  singler_prediction = as.character(singler_preds),
  sctype_prediction = if (!is.null(sctype_preds)) as.character(sctype_preds[ref_cluster_labels]) else NA,
  stringsAsFactors = FALSE
)

out_file <- file.path(output_dir, paste0(gsub("[^A-Za-z0-9_()-]", "_", ds_name), "_predictions.csv"))
write.csv(pred_df, out_file, row.names = FALSE)
cat(sprintf("  Saved: %s (%d clusters)\n", out_file, nrow(pred_df)))

# Quick accuracy (exact match, case-insensitive)
singler_acc <- mean(tolower(singler_preds) == tolower(names(singler_preds)), na.rm = TRUE)
cat(sprintf("  SingleR exact-match accuracy: %.3f\n", singler_acc))
if (!is.null(sctype_preds)) {
  sctype_acc <- mean(tolower(sctype_preds) == tolower(names(sctype_preds)), na.rm = TRUE)
  cat(sprintf("  scType exact-match accuracy: %.3f\n", sctype_acc))
}

cat(sprintf("  Done: %s\n", ds_name))
