# Preprocessing script for HLCA leiden 2 cell type benchmark
library(dplyr)
library(tidyr)
library(readr)

# Set paths
project_root <- "."

# Read both marker files
vs_sisters_raw <- read.csv(file.path(project_root, "data/reference/HLCA_Cluster_leiden_2_markers_vs_sisters.csv"), header = FALSE)
vs_all_raw <- read.csv(file.path(project_root, "data/reference/HLCA_Cluster_leiden_2_markers_vs_all.csv"), header = FALSE)

# Initialize list to store markers for each cell type
markers_list <- list()

# Process each cluster
unique_clusters <- unique(c(as.character(vs_all_raw[,1]), as.character(vs_sisters_raw[,1])))

for (i in seq_along(unique_clusters)) {
  cluster <- unique_clusters[i]
  # Get genes from vs_all
  all_row <- which(vs_all_raw[,1] == cluster)
  vs_all_genes <- character(0)
  if (length(all_row) > 0) {
    vs_all_genes <- vs_all_raw[all_row[1], -1]
    vs_all_genes <- unique(vs_all_genes[vs_all_genes != ""])
  }

  # Get genes from vs_sisters
  sisters_row <- which(vs_sisters_raw[,1] == cluster)
  vs_sisters_genes <- character(0)
  if (length(sisters_row) > 0) {
    vs_sisters_genes <- vs_sisters_raw[sisters_row[1], -1]
    vs_sisters_genes <- unique(vs_sisters_genes[vs_sisters_genes != ""])
  }

  # Combine results with labels
  combined_genes <- c(
    "The following markers are from comparison with all clusters:",
    vs_all_genes,
    "The following markers are from comparison with sister clusters:",
    vs_sisters_genes
  )

  # Store in list if we have genes
  if (length(combined_genes) > 0) {
    markers_list[[cluster]] <- list(genes = combined_genes)
  }
}

# Print data structure for debugging
print("Processed data structure:")
str(markers_list)
print("\nNumber of clusters:")
print(length(markers_list))

# Print first cluster genes for verification
print("\nFirst cluster genes:")
print(markers_list[[1]]$genes)

# Create results directory if it doesn't exist
dir.create(file.path(project_root, "results/benchmark/popv_comparison/1_preprocessing"), recursive = TRUE, showWarnings = FALSE)

# Save preprocessed data
saveRDS(markers_list, file.path(project_root, "results/benchmark/popv_comparison/1_preprocessing/HLCA_leiden_2_markers.rds"))

print("\nPreprocessing complete. Data saved to HLCA_leiden_2_markers.rds")
