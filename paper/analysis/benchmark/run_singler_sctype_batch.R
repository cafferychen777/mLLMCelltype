#!/usr/bin/env Rscript
# Batch SingleR & scType evaluation on all 49 benchmark datasets.
# Outputs per-dataset accuracy CSV for manuscript Figure/Extended Data.
#
# Usage: Rscript run_singler_sctype_batch.R

suppressPackageStartupMessages({
  library(SingleR)
  library(celldex)
  library(anndata)
  library(SingleCellExperiment)
  library(scuttle)
  library(scran)
  library(Matrix)
  library(HGNChelper)
  library(openxlsx)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR <- "."
DATA_DIR <- file.path(BASE_DIR, "data/raw")
RESULTS_DIR <- file.path(BASE_DIR, "results/benchmark/singler_sctype_batch")
SUPP_DATA <- file.path(BASE_DIR, "manuscript/Supplementary_Data_1.xlsx")

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# scType helper functions (from official GitHub)
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

# ---------------------------------------------------------------------------
# Dataset configuration: h5ad path, label column, tissue type for scType
# ---------------------------------------------------------------------------
DATASETS <- list(
  # Tabula Sapiens datasets
  list(name="Bladder(TS)", file="TS_Bladder_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Blood(TS)", file="TS_Blood_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Bone Marrow(TS)", file="TS_Bone_Marrow_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Eye(TS)", file="TS_Eye_filtered.h5ad", label="cell_ontology_class", tissue="Eye"),
  list(name="Fat(TS)", file="TS_Fat_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Heart(TS)", file="TS_Heart_filtered.h5ad", label="cell_ontology_class", tissue="Heart"),
  list(name="Large Intestine(TS)", file="TS_Large_Intestine_filtered.h5ad", label="cell_ontology_class", tissue="Intestine"),
  list(name="Liver(TS)", file="TS_Liver_filtered.h5ad", label="cell_ontology_class", tissue="Liver"),
  list(name="Lung(TS)", file="TS_Lung_filtered.h5ad", label="cell_ontology_class", tissue="Lung"),
  list(name="Lymph Node(TS)", file="TS_Lymph_Node_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Mammary(TS)", file="TS_Mammary_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Muscle(TS)", file="TS_Muscle_filtered.h5ad", label="cell_ontology_class", tissue="Skeletal muscle"),
  list(name="Pancreas(TS)", file="TS_Pancreas_filtered.h5ad", label="cell_ontology_class", tissue="Pancreas"),
  list(name="Prostate(TS)", file="TS_Prostate_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Salivary Gland(TS)", file="TS_Salivary_Gland_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Skin(TS)", file="TS_Skin_filtered.h5ad", label="cell_ontology_class", tissue="Skin"),
  list(name="Small Intestine(TS)", file="TS_Small_Intestine_filtered.h5ad", label="cell_ontology_class", tissue="Intestine"),
  list(name="Spleen(TS)", file="TS_Spleen_filtered.h5ad", label="cell_ontology_class", tissue="Spleen"),
  list(name="Thymus(TS)", file="TS_Thymus_filtered.h5ad", label="cell_ontology_class", tissue="Thymus"),
  list(name="Tongue(TS)", file="TS_Tongue_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Trachea(TS)", file="TS_Trachea_filtered.h5ad", label="cell_ontology_class", tissue="Lung"),
  list(name="Uterus(TS)", file="TS_Uterus_filtered.h5ad", label="cell_ontology_class", tissue="Immune system"),
  list(name="Vasculature(TS)", file="TS_Vasculature_filtered.h5ad", label="cell_ontology_class", tissue="Immune system")
)

# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
cat("Loading SingleR references...\n")
ref_hpca <- HumanPrimaryCellAtlasData()

# scType DB
DB_URL <- "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

# ---------------------------------------------------------------------------
# Helper: run SingleR on an SCE, return cluster-level predictions
# ---------------------------------------------------------------------------
run_singler_on_dataset <- function(sce, clusters) {
  # SingleR predicts per cell
  pred <- SingleR(
    test = sce,
    ref = ref_hpca,
    labels = ref_hpca$label.main,
    de.method = "wilcox"
  )
  # Aggregate to cluster level: majority vote
  cell_labels <- pred$labels
  cluster_labels <- tapply(cell_labels, clusters, function(x) {
    tab <- table(x)
    names(tab)[which.max(tab)]
  })
  return(cluster_labels)
}

# ---------------------------------------------------------------------------
# Helper: run scType on scaled matrix, return cluster-level predictions
# ---------------------------------------------------------------------------
run_sctype_on_dataset <- function(scaled_matrix, clusters, tissue) {
  # Prepare gene sets
  gs_list <- tryCatch(
    gene_sets_prepare(DB_URL, tissue),
    error = function(e) {
      cat("  scType gene set error for tissue:", tissue, "-", e$message, "\n")
      return(NULL)
    }
  )
  if (is.null(gs_list)) return(NULL)

  # Score
  es.max <- sctype_score(
    scRNAseqData = scaled_matrix,
    scaled = TRUE,
    gs = gs_list$gs_positive,
    gs2 = gs_list$gs_negative
  )

  # Assign per cluster
  unique_clusters <- sort(unique(clusters))
  cluster_labels <- character(length(unique_clusters))
  names(cluster_labels) <- unique_clusters

  for (cl in unique_clusters) {
    cell_idx <- which(clusters == cl)
    if (length(cell_idx) == 0) next
    scores <- rowMeans(es.max[, cell_idx, drop = FALSE])
    best <- names(which.max(scores))
    n_cells <- length(cell_idx)
    if (max(scores) < n_cells / 4) {
      cluster_labels[cl] <- "Unknown"
    } else {
      cluster_labels[cl] <- best
    }
  }
  return(cluster_labels)
}

# ---------------------------------------------------------------------------
# Helper: simple accuracy (exact string match, case-insensitive)
# ---------------------------------------------------------------------------
calculate_accuracy <- function(predicted, reference) {
  pred_lower <- tolower(predicted)
  ref_lower <- tolower(reference)
  mean(pred_lower == ref_lower, na.rm = TRUE)
}

# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
all_results <- data.frame(
  dataset = character(),
  singler_accuracy = numeric(),
  sctype_accuracy = numeric(),
  n_clusters = integer(),
  n_cells = integer(),
  stringsAsFactors = FALSE
)

for (ds in DATASETS) {
  h5ad_path <- file.path(DATA_DIR, ds$file)

  if (!file.exists(h5ad_path)) {
    cat(sprintf("SKIP: %s (file not found: %s)\n", ds$name, h5ad_path))
    next
  }

  cat(sprintf("\n===== Processing: %s =====\n", ds$name))

  tryCatch({
    # Load h5ad
    adata <- read_h5ad(h5ad_path)
    cat(sprintf("  Loaded: %d cells x %d genes\n", nrow(adata), ncol(adata)))

    # Get expression matrix and metadata (handle dgRMatrix → dgCMatrix)
    raw_mat <- adata$X
    if (!inherits(raw_mat, "dgCMatrix")) {
      raw_mat <- as(raw_mat, "CsparseMatrix")
    }
    expr_matrix <- as(t(raw_mat), "CsparseMatrix")  # genes x cells
    gene_names <- adata$var_names
    cell_names <- adata$obs_names
    rownames(expr_matrix) <- gene_names
    colnames(expr_matrix) <- cell_names

    # Get reference labels (factor → character)
    ref_labels <- as.character(adata$obs[[ds$label]])

    # Create cluster assignments (use reference labels as clusters for evaluation)
    clusters <- as.factor(ref_labels)

    # Create SCE for SingleR
    sce <- SingleCellExperiment(assays = list(counts = expr_matrix))
    sce <- computeSumFactors(sce)
    sce <- logNormCounts(sce)

    # --- SingleR ---
    cat("  Running SingleR...\n")
    singler_cluster_preds <- run_singler_on_dataset(sce, clusters)

    # Get reference cluster labels (the cluster names ARE the reference labels)
    ref_cluster_labels <- names(singler_cluster_preds)
    singler_acc <- calculate_accuracy(singler_cluster_preds, ref_cluster_labels)
    cat(sprintf("  SingleR accuracy: %.3f\n", singler_acc))

    # --- scType ---
    cat("  Running scType...\n")
    # Scale the log-normalized data
    log_counts <- as.matrix(logcounts(sce))
    scaled_matrix <- t(scale(t(log_counts)))
    scaled_matrix[is.nan(scaled_matrix)] <- 0

    sctype_cluster_preds <- run_sctype_on_dataset(scaled_matrix, clusters, ds$tissue)
    if (!is.null(sctype_cluster_preds)) {
      sctype_acc <- calculate_accuracy(sctype_cluster_preds, names(sctype_cluster_preds))
      cat(sprintf("  scType accuracy: %.3f\n", sctype_acc))
    } else {
      sctype_acc <- NA
      cat("  scType: FAILED\n")
    }

    # Save per-dataset predictions
    pred_df <- data.frame(
      cluster = ref_cluster_labels,
      reference = ref_cluster_labels,
      singler_prediction = as.character(singler_cluster_preds),
      sctype_prediction = if (!is.null(sctype_cluster_preds)) as.character(sctype_cluster_preds[ref_cluster_labels]) else NA,
      stringsAsFactors = FALSE
    )
    write.csv(pred_df, file.path(RESULTS_DIR, paste0(ds$name, "_predictions.csv")), row.names = FALSE)

    # Append to results
    all_results <- rbind(all_results, data.frame(
      dataset = ds$name,
      singler_accuracy = singler_acc,
      sctype_accuracy = sctype_acc,
      n_clusters = length(unique(clusters)),
      n_cells = ncol(expr_matrix),
      stringsAsFactors = FALSE
    ))

    cat(sprintf("  Done: %s (SingleR=%.1f%%, scType=%.1f%%)\n",
                ds$name, singler_acc * 100, sctype_acc * 100))

    # Save incremental results after each dataset
    write.csv(all_results, file.path(RESULTS_DIR, "singler_sctype_per_dataset_results.csv"), row.names = FALSE)

    # Clean up memory
    rm(adata, expr_matrix, sce, log_counts, scaled_matrix)
    gc()

  }, error = function(e) {
    cat(sprintf("  ERROR on %s: %s\n", ds$name, e$message))
  })
}

# ---------------------------------------------------------------------------
# Save summary results
# ---------------------------------------------------------------------------
output_path <- file.path(RESULTS_DIR, "singler_sctype_per_dataset_results.csv")
write.csv(all_results, output_path, row.names = FALSE)
cat(sprintf("\n\nResults saved to: %s\n", output_path))
cat(sprintf("Processed %d datasets\n", nrow(all_results)))
cat(sprintf("Mean SingleR accuracy: %.1f%%\n", mean(all_results$singler_accuracy, na.rm = TRUE) * 100))
cat(sprintf("Mean scType accuracy: %.1f%%\n", mean(all_results$sctype_accuracy, na.rm = TRUE) * 100))
