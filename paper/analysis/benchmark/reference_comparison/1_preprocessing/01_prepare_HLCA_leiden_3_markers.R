# Preprocessing script for HLCA leiden 3 cell type benchmark
library(dplyr)
library(tidyr)
library(readr)

# Function to get Level 2 parent from cluster ID
get_l2_parent <- function(cluster_id) {
  # Extract the first two numbers from the cluster ID
  # e.g., "1.2.3" -> "1.2"
  parts <- strsplit(cluster_id, "\\.")[[1]]
  if (length(parts) >= 2) {
    # If it's a pattern like "1.0.0", return "1"
    if (parts[2] == "0") {
      return(parts[1])
    }
    # Otherwise return first two parts (e.g., "1.2")
    return(paste(parts[1:2], collapse = "."))
  }
  return(parts[1])  # For single number cases
}

# Set paths
project_root <- "."

# Read both marker files
vs_sisters_raw <- read.csv(file.path(project_root, "data/reference/HLCA_Cluster_leiden_3_markers_vs_sisters.csv"), header = FALSE)
vs_all_raw <- read.csv(file.path(project_root, "data/reference/HLCA_Cluster_leiden_3_markers_vs_all.csv"), header = FALSE)

# Read Level 2 consensus results and existing markers
level2_results <- readRDS(file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_2_results.rds"))
existing_markers <- readRDS(file.path(project_root, "results/benchmark/popv_comparison/1_preprocessing/HLCA_leiden_2_markers.rds"))

# Extract Level 2 annotations from final_annotations
level2_consensus <- level2_results$final_annotations

print("Level 2 consensus structure:")
print(str(level2_consensus))

# Get the order of Level 2 IDs from existing markers
level2_ids <- names(existing_markers)
print("Level 2 IDs order:")
print(level2_ids)

# Create mapping using Level 2 IDs order
level2_map <- list()
for (i in seq_along(level2_ids)) {
  # Get the corresponding annotation from level2_consensus using the index
  annotation <- level2_consensus[[i]]
  # If annotation is a named vector, get just the value
  if (length(names(annotation)) > 0) {
    annotation <- unname(annotation)
  }
  # Store using the Level 2 ID
  level2_map[[level2_ids[i]]] <- annotation
}

print("Level 2 annotations after mapping:")
print(str(level2_map))

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

  # Get Level 2 parent and annotation
  l2_parent <- get_l2_parent(cluster)
  print(sprintf("Processing cluster %s, L2 parent: %s", cluster, l2_parent))
  l2_type <- if (!is.null(l2_parent)) level2_map[[l2_parent]] else "Unknown"
  print(sprintf("L2 type: %s", l2_type))

  # Combine results with labels
  combined_genes <- c(
    "The following markers are from comparison with all clusters:",
    vs_all_genes,
    if(length(vs_sisters_genes) > 0) c(
      "The following markers are from comparison with sister clusters:",
      vs_sisters_genes
    ),
    sprintf("This cluster was annotated as '%s' in the Level 2 analysis. Please provide a more specific annotation within this cell type category.", l2_type)
  )

  # Store in list with proper structure
  markers_list[[cluster]] <- list(genes = combined_genes)
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
saveRDS(markers_list, file.path(project_root, "results/benchmark/popv_comparison/1_preprocessing/HLCA_leiden_3_markers.rds"))

print("\nPreprocessing complete. Data saved to HLCA_leiden_3_markers.rds")
