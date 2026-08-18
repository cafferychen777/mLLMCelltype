#!/usr/bin/env Rscript
# Re-run scType ONLY on pre-processed datasets, keeping existing SingleR predictions.
# Fixes the rowMeans→rowSums bug and updates sctype_prediction in existing CSVs.
#
# Memory-optimized: only normalizes/scales genes in the scType marker gene sets,
# avoiding dense matrix expansion of the full expression matrix.
#
# Usage:
#   Rscript rerun_sctype_only.R --prepared_dir DIR --results_dir DIR [--dataset NAME]

suppressPackageStartupMessages({
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
  stop("Usage: Rscript rerun_sctype_only.R --prepared_dir DIR --results_dir DIR [--dataset NAME]")
}

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

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
# Helper: get marker genes from scType gene sets for a tissue
# Returns the union of all positive and negative marker genes
# ---------------------------------------------------------------------------
get_marker_genes <- function(gs_list) {
  pos_genes <- unique(unlist(gs_list$gs_positive))
  neg_genes <- unique(unlist(gs_list$gs_negative))
  unique(c(pos_genes, neg_genes))
}

# ---------------------------------------------------------------------------
# Helper: run scType with FIXED rowSums (matching official tutorial)
# Memory-optimized: only processes marker genes
# ---------------------------------------------------------------------------
run_sctype_on_dataset <- function(counts_matrix, clusters, tissue) {
  # Prepare gene sets first (before converting to dense)
  gs_list <- tryCatch(
    gene_sets_prepare(DB_URL, tissue),
    error = function(e) {
      cat("  scType gene set error for tissue:", tissue, "-", e$message, "\n")
      return(NULL)
    }
  )
  if (is.null(gs_list)) return(NULL)

  # Get marker genes and subset
  marker_genes <- get_marker_genes(gs_list)
  # Also try uppercase versions
  all_genes <- rownames(counts_matrix)
  genes_upper <- toupper(all_genes)

  # Find which marker genes are in the expression matrix
  gene_idx <- which(all_genes %in% marker_genes | genes_upper %in% toupper(marker_genes))
  cat(sprintf("  Marker genes: %d total, %d found in matrix\n",
              length(marker_genes), length(gene_idx)))

  if (length(gene_idx) == 0) {
    cat("  WARNING: No marker genes found in expression matrix!\n")
    return(NULL)
  }

  # Subset to marker genes only (keep sparse until this point)
  sub_counts <- counts_matrix[gene_idx, , drop = FALSE]

  # Log-normalize (Seurat-style: divide by library size, scale by 10000, log1p)
  # Use full matrix library sizes for proper normalization
  lib_sizes <- colSums(counts_matrix)
  lib_sizes[lib_sizes == 0] <- 1
  sub_norm <- t(t(sub_counts) / lib_sizes) * 10000
  sub_log <- log1p(as.matrix(sub_norm))

  # Scale genes (z-score per gene across cells)
  scaled_sub <- t(scale(t(sub_log)))
  scaled_sub[is.nan(scaled_sub)] <- 0

  rm(sub_counts, sub_norm, sub_log)
  gc()

  # Now we need to provide a full-sized matrix for sctype_score
  # but only the marker gene rows need values; rest can be zeros
  # Actually, sctype_score subsets internally, so we can just provide the subset
  # But row names must match the marker gene names in the gene set

  es.max <- sctype_score(
    scRNAseqData = scaled_sub,
    scaled = TRUE,
    gs = gs_list$gs_positive,
    gs2 = gs_list$gs_negative
  )

  rm(scaled_sub)
  gc()

  unique_clusters <- sort(unique(clusters))
  cluster_labels <- character(length(unique_clusters))
  names(cluster_labels) <- unique_clusters

  for (cl in unique_clusters) {
    cell_idx <- which(clusters == cl)
    if (length(cell_idx) == 0) next
    # FIXED: use rowSums (not rowMeans) to match official scType tutorial
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

cat(sprintf("Found %d dataset(s) to process (scType-only rerun)\n", length(all_dirs)))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
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

  # Check if existing predictions CSV exists (to keep SingleR results)
  pred_csv <- file.path(results_dir, paste0(ds_name, "_predictions.csv"))
  if (!file.exists(pred_csv)) {
    cat(sprintf("SKIP %s: no existing predictions CSV found\n", ds_name))
    next
  }

  cat(sprintf("\n===== scType rerun: %s (%s, tissue=%s) =====\n", display_name, ds_name, tissue))

  tryCatch({
    # Load expression matrix (genes x cells, sparse)
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

    # Run scType (memory-optimized: only processes marker genes)
    sctype_preds <- run_sctype_on_dataset(expr_matrix, clusters, tissue)

    rm(expr_matrix)
    gc()

    if (!is.null(sctype_preds)) {
      sctype_acc <- calculate_accuracy(sctype_preds, names(sctype_preds))
      cat(sprintf("  scType accuracy: %.3f\n", sctype_acc))
    } else {
      sctype_acc <- NA
      cat("  scType: FAILED\n")
    }

    # Read existing predictions (to preserve SingleR results)
    existing_pred <- read.csv(pred_csv, stringsAsFactors = FALSE)

    # Update sctype_prediction column
    if (!is.null(sctype_preds)) {
      existing_pred$sctype_prediction <- as.character(sctype_preds[existing_pred$cluster])
    }

    # Save updated predictions
    write.csv(existing_pred, pred_csv, row.names = FALSE)
    cat(sprintf("  Updated: %s (scType=%.1f%%)\n", pred_csv, sctype_acc * 100))

    # Show prediction distribution
    pred_table <- table(existing_pred$sctype_prediction)
    n_unknown <- sum(pred_table[names(pred_table) == "Unknown"])
    cat(sprintf("  Predictions: %d Unknown / %d total\n", n_unknown, nrow(existing_pred)))

    gc()

  }, error = function(e) {
    cat(sprintf("  ERROR on %s: %s\n", display_name, e$message))
  })
}

cat("\n\nscType-only rerun complete!\n")
