#!/usr/bin/env Rscript
# Download Azimuth reference datasets and prepare for SingleR/scType.
# Uses direct slot access to avoid Seurat loading issues.
#
# Usage:
#   Rscript prepare_azimuth_data_v2.R --output_dir DIR

suppressPackageStartupMessages({
  library(Matrix)
  library(SeuratObject)
})

# Extra lib path for packages installed to scratch
scratch_root <- Sys.getenv(
  "MLLMCELLTYPE_SCRATCH_ROOT",
  file.path("/scratch/user", Sys.info()[["user"]])
)
EXTRA_LIB <- Sys.getenv(
  "MLLMCELLTYPE_R_LIB",
  file.path(scratch_root, "R_libs")
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
output_dir <- NULL

i <- 1
while (i <= length(args)) {
  if (args[i] == "--output_dir") {
    output_dir <- args[i + 1]; i <- i + 2
  } else {
    i <- i + 1
  }
}

if (is.null(output_dir)) {
  stop("Usage: Rscript prepare_azimuth_data_v2.R --output_dir DIR")
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Azimuth reference configs
# Use first label column that has a reasonable number of types for benchmarking
AZIMUTH_REFS <- list(
  adiposeref = list(
    dir = "Azimuth_adipose", name = "Azimuth Adipose",
    label_col = "celltype.l2", tissue = "Immune system"
  ),
  bonemarrowref = list(
    dir = "Azimuth_bone_marrow", name = "Azimuth Bone Marrow",
    label_col = "celltype.l2", tissue = "Immune system"
  ),
  fetusref = list(
    dir = "Azimuth_fetal", name = "Azimuth Fetal Development",
    label_col = "annotation.l1", tissue = "Immune system"
  ),
  heartref = list(
    dir = "Azimuth_heart", name = "Azimuth Heart",
    label_col = "celltype.l2", tissue = "Heart"
  ),
  kidneyref = list(
    dir = "Azimuth_kidney", name = "Azimuth Kidney",
    label_col = "annotation.l1", tissue = "Kidney"
  ),
  lungref = list(
    dir = "Azimuth_lung", name = "Azimuth Lung",
    label_col = "ann_level_2", tissue = "Lung"
  ),
  pancreasref = list(
    dir = "Azimuth_pancreas", name = "Azimuth Pancreas",
    label_col = "annotation.l1", tissue = "Pancreas"
  ),
  tonsilref = list(
    dir = "Azimuth_tonsil", name = "Azimuth Tonsil",
    label_col = "celltype.l2", tissue = "Immune system"
  )
)

# ---------------------------------------------------------------------------
# Process one Azimuth reference
# ---------------------------------------------------------------------------
process_azimuth_ref <- function(pkg_name, config) {
  ds_dir <- file.path(output_dir, config$dir)

  if (file.exists(file.path(ds_dir, "expression.mtx"))) {
    cat(sprintf("  Skipping %s: already processed\n", config$name))
    return(TRUE)
  }

  dir.create(ds_dir, showWarnings = FALSE, recursive = TRUE)

  tryCatch({
    # Find the RDS file
    full_pkg <- paste0(pkg_name, ".SeuratData")
    rds_path <- system.file("azimuth", "ref.Rds", package = full_pkg,
                            lib.loc = c(.libPaths(), EXTRA_LIB))
    if (rds_path == "") {
      cat(sprintf("  ERROR: Package %s not installed\n", full_pkg))
      return(FALSE)
    }

    cat(sprintf("  Reading %s...\n", rds_path))
    ref <- readRDS(rds_path)

    # Access metadata directly
    meta <- ref@meta.data
    n_cells <- nrow(meta)
    cat(sprintf("  %d cells, metadata cols: %s\n", n_cells,
                paste(colnames(meta), collapse = ", ")))

    # Find label column
    label_col <- config$label_col
    if (!(label_col %in% colnames(meta))) {
      avail <- colnames(meta)
      alternatives <- c("celltype.l1", "celltype.l2", "annotation.l1",
                         "ann_level_1", "ann_level_2", "cell_type")
      found <- FALSE
      for (alt in alternatives) {
        if (alt %in% avail) {
          label_col <- alt
          found <- TRUE
          break
        }
      }
      if (!found) {
        cat(sprintf("  ERROR: No suitable label column. Available: %s\n",
                    paste(avail, collapse = ", ")))
        return(FALSE)
      }
    }

    labels <- as.character(meta[[label_col]])
    n_clusters <- length(unique(labels))
    cat(sprintf("  Label column: %s (%d unique types)\n", label_col, n_clusters))

    # Get expression data via direct slot access
    assays_slot <- slot(ref, "assays")
    assay_obj <- assays_slot[[1]]  # First (and usually only) assay
    expr <- slot(assay_obj, "data")

    if (is.null(expr) || prod(dim(expr)) == 0) {
      # Try counts
      expr <- slot(assay_obj, "counts")
    }

    if (is.null(expr) || prod(dim(expr)) == 0) {
      cat("  ERROR: Could not extract expression data\n")
      return(FALSE)
    }

    cat(sprintf("  Expression: %d genes x %d cells\n", nrow(expr), ncol(expr)))

    # Ensure sparse
    if (!inherits(expr, "dgCMatrix")) {
      expr <- as(expr, "dgCMatrix")
    }

    # Save
    writeMM(expr, file.path(ds_dir, "expression.mtx"))
    write.csv(data.frame(gene = rownames(expr)),
              file.path(ds_dir, "genes.csv"), row.names = FALSE)
    write.csv(data.frame(barcode = colnames(expr)),
              file.path(ds_dir, "barcodes.csv"), row.names = FALSE)
    write.csv(data.frame(barcode = colnames(expr), label = labels),
              file.path(ds_dir, "metadata.csv"), row.names = FALSE)
    writeLines(c(
      sprintf("name=%s", config$name),
      sprintf("tissue=%s", config$tissue),
      sprintf("n_cells=%d", ncol(expr)),
      sprintf("n_genes=%d", nrow(expr)),
      sprintf("label_col=%s", label_col),
      sprintf("n_clusters=%d", n_clusters)
    ), file.path(ds_dir, "config.txt"))

    cat(sprintf("  Saved to %s (%d cells, %d genes, %d clusters)\n",
                ds_dir, ncol(expr), nrow(expr), n_clusters))

    rm(ref, expr, assay_obj, assays_slot)
    gc()
    return(TRUE)

  }, error = function(e) {
    cat(sprintf("  ERROR processing %s: %s\n", config$name, e$message))
    return(FALSE)
  })
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
cat(sprintf("Processing %d Azimuth reference(s)\n", length(AZIMUTH_REFS)))

success <- 0
failed <- 0
for (pkg_name in names(AZIMUTH_REFS)) {
  config <- AZIMUTH_REFS[[pkg_name]]
  cat(sprintf("\n===== %s (%s) =====\n", config$name, pkg_name))
  if (process_azimuth_ref(pkg_name, config)) {
    success <- success + 1
  } else {
    failed <- failed + 1
  }
}

cat(sprintf("\nDone. Success: %d, Failed: %d\n", success, failed))
