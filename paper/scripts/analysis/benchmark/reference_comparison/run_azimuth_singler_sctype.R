#!/usr/bin/env Rscript
# Run SingleR and scType on Azimuth + MCA datasets.
# Each prepared dataset directory contains:
#   expression.mtx, genes.csv, barcodes.csv, metadata.csv, config.txt
#
# Usage:
#   Rscript run_azimuth_singler_sctype.R \
#     --prepared_dir /path/to/prepared \
#     --results_dir /path/to/results \
#     [--dataset Azimuth_adipose]

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

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)
prepared_dir <- NULL
results_dir <- NULL
target_dataset <- "all"

i <- 1
while (i <= length(args)) {
  if (args[i] == "--prepared_dir") {
    prepared_dir <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--results_dir") {
    results_dir <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--dataset") {
    target_dataset <- args[i + 1]; i <- i + 2
  } else {
    i <- i + 1
  }
}

if (is.null(prepared_dir) || is.null(results_dir)) {
  stop("Usage: Rscript run_azimuth_singler_sctype.R --prepared_dir DIR --results_dir DIR [--dataset NAME]")
}

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# Load SingleR references
cat("Loading SingleR references...\n")
ref_hpca <- HumanPrimaryCellAtlasData()
cat("  Human reference loaded:", ncol(ref_hpca), "cells\n")

# Also load mouse reference for MCA
tryCatch({
  ref_mouse <- MouseRNAseqData()
  cat("  Mouse reference loaded:", ncol(ref_mouse), "cells\n")
}, error = function(e) {
  cat("  Warning: Could not load MouseRNAseqData:", e$message, "\n")
  ref_mouse <<- NULL
})

# scType DB
DB_URL <- "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

# Helper functions
read_config <- function(config_path) {
  lines <- readLines(config_path)
  config <- list()
  for (line in lines) {
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    if (length(parts) == 2) {
      config[[parts[1]]] <- parts[2]
    }
  }
  return(config)
}

run_singler_on_dataset <- function(sce, clusters, ref) {
  pred <- SingleR(
    test = sce,
    ref = ref,
    labels = ref$label.main,
    de.method = "wilcox"
  )
  cell_labels <- pred$labels
  cluster_labels <- tapply(cell_labels, clusters, function(x) {
    tab <- table(x)
    names(tab)[which.max(tab)]
  })
  return(cluster_labels)
}

run_sctype_on_dataset <- function(scaled_matrix, clusters, tissue) {
  tryCatch({
    gs_list <- gene_sets_prepare(DB_URL, tissue)
    es_max <- sctype_score(
      scRNAseqData = scaled_matrix,
      scaled = TRUE,
      gs = gs_list$gs_positive,
      gs2 = gs_list$gs_negative
    )
    # Get cluster-level scores
    unique_clusters <- levels(clusters)
    if (is.null(unique_clusters)) unique_clusters <- unique(clusters)

    preds <- character(length(unique_clusters))
    names(preds) <- unique_clusters

    for (cl in unique_clusters) {
      cl_cells <- which(clusters == cl)
      if (length(cl_cells) > 0) {
        cl_scores <- rowSums(es_max[, cl_cells, drop = FALSE])
        preds[cl] <- names(which.max(cl_scores))
      }
    }
    return(preds)
  }, error = function(e) {
    cat(sprintf("  scType error: %s\n", e$message))
    return(NULL)
  })
}

calculate_accuracy <- function(predictions, references) {
  matches <- sum(tolower(predictions) == tolower(references))
  return(matches / length(references))
}

# Process datasets
ds_dirs <- list.dirs(prepared_dir, recursive = FALSE, full.names = FALSE)
if (target_dataset != "all") {
  ds_dirs <- ds_dirs[ds_dirs == target_dataset]
}

all_results <- data.frame()

for (ds_name in ds_dirs) {
  ds_dir <- file.path(prepared_dir, ds_name)

  required_files <- c("expression.mtx", "genes.csv", "barcodes.csv", "metadata.csv", "config.txt")
  missing <- required_files[!file.exists(file.path(ds_dir, required_files))]
  if (length(missing) > 0) {
    cat(sprintf("SKIP %s: missing files: %s\n", ds_name, paste(missing, collapse = ", ")))
    next
  }

  config <- read_config(file.path(ds_dir, "config.txt"))
  display_name <- config$name
  tissue <- config$tissue
  is_mouse <- !is.null(config$organism) && config$organism == "mouse"

  cat(sprintf("\n===== Processing: %s (%s) =====\n", display_name, ds_name))
  if (is_mouse) cat("  [Mouse data - using MouseRNAseqData reference]\n")

  tryCatch({
    # Load expression matrix
    cat("  Loading expression matrix...\n")
    expr_matrix <- readMM(file.path(ds_dir, "expression.mtx"))
    expr_matrix <- as(expr_matrix, "dgCMatrix")

    genes <- read.csv(file.path(ds_dir, "genes.csv"), stringsAsFactors = FALSE)$gene
    barcodes <- read.csv(file.path(ds_dir, "barcodes.csv"), stringsAsFactors = FALSE)$barcode
    metadata <- read.csv(file.path(ds_dir, "metadata.csv"), stringsAsFactors = FALSE)

    rownames(expr_matrix) <- genes
    colnames(expr_matrix) <- barcodes
    cat(sprintf("  Matrix: %d genes x %d cells\n", nrow(expr_matrix), ncol(expr_matrix)))

    # Create clusters from full data
    ref_labels <- metadata$label
    clusters_full <- as.factor(ref_labels)
    cat(sprintf("  Cell types: %d\n", nlevels(clusters_full)))

    # Subsample to max 500 cells per cluster for memory efficiency
    max_per_cluster <- 500
    set.seed(42)
    keep_idx <- c()
    for (cl in levels(clusters_full)) {
      cl_idx <- which(clusters_full == cl)
      if (length(cl_idx) > max_per_cluster) {
        cl_idx <- sample(cl_idx, max_per_cluster)
      }
      keep_idx <- c(keep_idx, cl_idx)
    }
    keep_idx <- sort(keep_idx)
    cat(sprintf("  Subsampled %d -> %d cells (max %d per cluster)\n",
                ncol(expr_matrix), length(keep_idx), max_per_cluster))
    expr_sub <- expr_matrix[, keep_idx]
    clusters <- clusters_full[keep_idx]
    rm(expr_matrix)
    gc()

    # Create SCE from subsampled data
    sce <- SingleCellExperiment(assays = list(counts = expr_sub))
    cat("  Normalizing (library size factors)...\n")
    sce <- logNormCounts(sce)
    rm(expr_sub)
    gc()

    # --- SingleR ---
    cat("  Running SingleR...\n")
    singler_ref <- if (is_mouse && !is.null(ref_mouse)) ref_mouse else ref_hpca
    singler_preds <- run_singler_on_dataset(sce, clusters, singler_ref)
    ref_cluster_labels <- names(singler_preds)
    singler_acc <- calculate_accuracy(singler_preds, ref_cluster_labels)
    cat(sprintf("  SingleR accuracy: %.3f\n", singler_acc))

    # --- scType ---
    sctype_preds <- NULL
    sctype_acc <- NA
    if (!is_mouse) {
      cat("  Running scType...\n")
      log_counts <- as.matrix(logcounts(sce))
      scaled_matrix <- t(scale(t(log_counts)))
      scaled_matrix[is.nan(scaled_matrix)] <- 0

      sctype_preds <- run_sctype_on_dataset(scaled_matrix, clusters, tissue)
      if (!is.null(sctype_preds)) {
        sctype_acc <- calculate_accuracy(sctype_preds, names(sctype_preds))
        cat(sprintf("  scType accuracy: %.3f\n", sctype_acc))
      } else {
        cat("  scType: FAILED\n")
      }
      rm(log_counts, scaled_matrix)
    } else {
      cat("  scType: SKIPPED (mouse data, human-only database)\n")
    }

    # Save predictions
    pred_df <- data.frame(
      cluster = ref_cluster_labels,
      reference = ref_cluster_labels,
      singler_prediction = as.character(singler_preds),
      sctype_prediction = if (!is.null(sctype_preds)) as.character(sctype_preds[ref_cluster_labels]) else NA,
      stringsAsFactors = FALSE
    )
    write.csv(pred_df, file.path(results_dir, paste0(ds_name, "_predictions.csv")), row.names = FALSE)

    all_results <- rbind(all_results, data.frame(
      dataset = display_name,
      singler_accuracy = singler_acc,
      sctype_accuracy = sctype_acc,
      n_clusters = nlevels(clusters),
      n_cells = ncol(sce),
      stringsAsFactors = FALSE
    ))

    cat(sprintf("  Done: %s (SingleR=%.1f%%, scType=%s)\n",
                display_name, singler_acc * 100,
                if (is.na(sctype_acc)) "N/A" else sprintf("%.1f%%", sctype_acc * 100)))

    rm(sce)
    gc()

  }, error = function(e) {
    cat(sprintf("  ERROR on %s: %s\n", display_name, e$message))
  })
}

# Save summary
output_path <- file.path(results_dir, "azimuth_singler_sctype_results.csv")
write.csv(all_results, output_path, row.names = FALSE)
cat(sprintf("\n\nResults saved to: %s\n", output_path))
cat(sprintf("Processed %d datasets\n", nrow(all_results)))
if (nrow(all_results) > 0) {
  cat(sprintf("Mean SingleR accuracy: %.1f%%\n", mean(all_results$singler_accuracy, na.rm = TRUE) * 100))
  human_results <- all_results[!is.na(all_results$sctype_accuracy), ]
  if (nrow(human_results) > 0) {
    cat(sprintf("Mean scType accuracy: %.1f%%\n", mean(human_results$sctype_accuracy, na.rm = TRUE) * 100))
  }
}
