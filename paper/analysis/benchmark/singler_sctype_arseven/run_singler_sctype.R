#!/usr/bin/env Rscript
# Run SingleR and scType on pre-processed datasets (prepared by prepare_data.py).
# Each dataset directory contains: expression.mtx, genes.csv, barcodes.csv, metadata.csv, config.txt
#
# Usage:
#   Rscript run_singler_sctype.R --prepared_dir /path/to/prepared --results_dir /path/to/results
#   Rscript run_singler_sctype.R --prepared_dir /path/to/prepared --results_dir /path/to/results --dataset BloodTS

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

# scType helper functions (from official GitHub)
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
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
  stop("Usage: Rscript run_singler_sctype.R --prepared_dir DIR --results_dir DIR [--dataset NAME]")
}

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Load SingleR reference
# ---------------------------------------------------------------------------
cat("Loading SingleR reference (HumanPrimaryCellAtlas)...\n")
ref_hpca <- HumanPrimaryCellAtlasData()
cat("  Reference loaded:", ncol(ref_hpca), "cells,", nrow(ref_hpca), "genes\n")

# scType DB
DB_URL <- "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

# ---------------------------------------------------------------------------
# Helper: read config file
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helper: run SingleR, return cluster-level predictions
# ---------------------------------------------------------------------------
run_singler_on_dataset <- function(sce, clusters) {
  pred <- SingleR(
    test = sce,
    ref = ref_hpca,
    labels = ref_hpca$label.main,
    de.method = "wilcox"
  )
  cell_labels <- pred$labels
  cluster_labels <- tapply(cell_labels, clusters, function(x) {
    tab <- table(x)
    names(tab)[which.max(tab)]
  })
  return(cluster_labels)
}

# ---------------------------------------------------------------------------
# Helper: run scType, return cluster-level predictions
# ---------------------------------------------------------------------------
run_sctype_on_dataset <- function(scaled_matrix, clusters, tissue) {
  gs_list <- tryCatch(
    gene_sets_prepare(DB_URL, tissue),
    error = function(e) {
      cat("  scType gene set error for tissue:", tissue, "-", e$message, "\n")
      return(NULL)
    }
  )
  if (is.null(gs_list)) return(NULL)

  es.max <- sctype_score(
    scRNAseqData = scaled_matrix,
    scaled = TRUE,
    gs = gs_list$gs_positive,
    gs2 = gs_list$gs_negative
  )

  unique_clusters <- sort(unique(clusters))
  cluster_labels <- character(length(unique_clusters))
  names(cluster_labels) <- unique_clusters

  for (cl in unique_clusters) {
    cell_idx <- which(clusters == cl)
    if (length(cell_idx) == 0) next
    scores <- rowSums(es.max[, cell_idx, drop = FALSE])
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
# Helper: accuracy
# ---------------------------------------------------------------------------
calculate_accuracy <- function(predicted, reference) {
  pred_lower <- tolower(predicted)
  ref_lower <- tolower(reference)
  mean(pred_lower == ref_lower, na.rm = TRUE)
}

# ---------------------------------------------------------------------------
# Find datasets
# ---------------------------------------------------------------------------
all_dirs <- list.dirs(prepared_dir, full.names = TRUE, recursive = FALSE)
if (target_dataset != "all") {
  all_dirs <- all_dirs[basename(all_dirs) == target_dataset]
  if (length(all_dirs) == 0) {
    stop("Dataset directory not found: ", target_dataset)
  }
}

cat(sprintf("Found %d dataset(s) to process\n", length(all_dirs)))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_results <- data.frame(
  dataset = character(),
  singler_accuracy = numeric(),
  sctype_accuracy = numeric(),
  n_clusters = integer(),
  n_cells = integer(),
  stringsAsFactors = FALSE
)

for (ds_dir in all_dirs) {
  ds_name <- basename(ds_dir)

  # Check required files
  required_files <- c("expression.mtx", "genes.csv", "barcodes.csv", "metadata.csv", "config.txt")
  missing <- required_files[!file.exists(file.path(ds_dir, required_files))]
  if (length(missing) > 0) {
    cat(sprintf("SKIP %s: missing files: %s\n", ds_name, paste(missing, collapse = ", ")))
    next
  }

  config <- read_config(file.path(ds_dir, "config.txt"))
  display_name <- config$name
  tissue <- config$tissue

  cat(sprintf("\n===== Processing: %s (%s) =====\n", display_name, ds_name))

  tryCatch({
    # Load expression matrix (genes x cells, already transposed by prepare_data.py)
    cat("  Loading expression matrix...\n")
    expr_matrix <- readMM(file.path(ds_dir, "expression.mtx"))
    expr_matrix <- as(expr_matrix, "dgCMatrix")

    genes <- read.csv(file.path(ds_dir, "genes.csv"), stringsAsFactors = FALSE)$gene
    barcodes <- read.csv(file.path(ds_dir, "barcodes.csv"), stringsAsFactors = FALSE)$barcode
    metadata <- read.csv(file.path(ds_dir, "metadata.csv"), stringsAsFactors = FALSE)

    rownames(expr_matrix) <- genes
    colnames(expr_matrix) <- barcodes
    cat(sprintf("  Matrix: %d genes x %d cells\n", nrow(expr_matrix), ncol(expr_matrix)))

    # Create clusters from reference labels
    ref_labels <- metadata$label
    clusters <- as.factor(ref_labels)

    # Create SCE
    sce <- SingleCellExperiment(assays = list(counts = expr_matrix))
    sce <- computeSumFactors(sce)
    sce <- logNormCounts(sce)

    # --- SingleR ---
    cat("  Running SingleR...\n")
    singler_preds <- run_singler_on_dataset(sce, clusters)
    ref_cluster_labels <- names(singler_preds)
    singler_acc <- calculate_accuracy(singler_preds, ref_cluster_labels)
    cat(sprintf("  SingleR accuracy: %.3f\n", singler_acc))

    # --- scType ---
    cat("  Running scType...\n")
    log_counts <- as.matrix(logcounts(sce))
    scaled_matrix <- t(scale(t(log_counts)))
    scaled_matrix[is.nan(scaled_matrix)] <- 0

    sctype_preds <- run_sctype_on_dataset(scaled_matrix, clusters, tissue)
    if (!is.null(sctype_preds)) {
      sctype_acc <- calculate_accuracy(sctype_preds, names(sctype_preds))
      cat(sprintf("  scType accuracy: %.3f\n", sctype_acc))
    } else {
      sctype_acc <- NA
      cat("  scType: FAILED\n")
    }

    # Save per-dataset predictions
    pred_df <- data.frame(
      cluster = ref_cluster_labels,
      reference = ref_cluster_labels,
      singler_prediction = as.character(singler_preds),
      sctype_prediction = if (!is.null(sctype_preds)) as.character(sctype_preds[ref_cluster_labels]) else NA,
      stringsAsFactors = FALSE
    )
    write.csv(pred_df, file.path(results_dir, paste0(ds_name, "_predictions.csv")), row.names = FALSE)

    # Append to results
    all_results <- rbind(all_results, data.frame(
      dataset = display_name,
      singler_accuracy = singler_acc,
      sctype_accuracy = sctype_acc,
      n_clusters = length(unique(clusters)),
      n_cells = ncol(expr_matrix),
      stringsAsFactors = FALSE
    ))

    cat(sprintf("  Done: %s (SingleR=%.1f%%, scType=%.1f%%)\n",
                display_name, singler_acc * 100, sctype_acc * 100))

    rm(expr_matrix, sce, log_counts, scaled_matrix)
    gc()

  }, error = function(e) {
    cat(sprintf("  ERROR on %s: %s\n", display_name, e$message))
  })
}

# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------
output_path <- file.path(results_dir, "singler_sctype_per_dataset_results.csv")
write.csv(all_results, output_path, row.names = FALSE)
cat(sprintf("\n\nResults saved to: %s\n", output_path))
cat(sprintf("Processed %d datasets\n", nrow(all_results)))
if (nrow(all_results) > 0) {
  cat(sprintf("Mean SingleR accuracy: %.1f%%\n", mean(all_results$singler_accuracy, na.rm = TRUE) * 100))
  cat(sprintf("Mean scType accuracy: %.1f%%\n", mean(all_results$sctype_accuracy, na.rm = TRUE) * 100))
}
