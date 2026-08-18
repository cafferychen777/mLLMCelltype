library(tidyverse)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Please provide a tissue name as argument")
}
tissue <- args[1]

# Set project root
project_root <- "."

# Function to get reference data for a tissue
get_reference_data <- function(tissue, project_root) {
  # Special case for kidney
  if (tissue == "kidney") {
    reference_data <- read_delim(
      file.path(project_root, "data/reference/kidney_markers_L1.csv"),
      delim = ",",
      col_names = c("celltype", "gene"),
      skip = 1
    ) %>%
      select(celltype) %>%
      slice(1:16) %>%
      pull(celltype)
    return(list(reference_data = reference_data, n_clusters = 16))
  }

  # For all other tissues, get reference names from the markers RDS file
  markers_path <- file.path(project_root,
                           "results/benchmark/reference_comparison/1_preprocessing",
                           paste0(tissue, "_markers.rds"))

  if (!file.exists(markers_path)) {
    stop(paste("Markers file not found for tissue:", tissue))
  }

  markers <- readRDS(markers_path)
  reference_data <- names(markers)
  n_clusters <- length(reference_data)

  return(list(reference_data = reference_data, n_clusters = n_clusters))
}

# Read the RDS file
results_path <- file.path(project_root,
                         "results/benchmark/reference_comparison/2_evaluation",
                         paste0(tissue, "_results.rds"))

if (!file.exists(results_path)) {
  stop(paste("Results file not found for tissue:", tissue))
}

results <- readRDS(results_path)

# Get reference data
ref_data <- get_reference_data(tissue, project_root)

# Create data frame with initial predictions
results_df <- tibble(
  cluster_id = as.character(1:ref_data$n_clusters),
  reference_name = ref_data$reference_data
) %>%
  mutate(
    final_consensus = sapply(cluster_id, function(id) {
      if (!is.null(results$final_annotations[[id]])) {
        results$final_annotations[[id]]
      } else {
        NA_character_
      }
    }),
    is_controversial = sapply(cluster_id, function(id) {
      !results$initial_results$consensus_results[[as.numeric(id)]]$reached
    }),
    agreement_score = sapply(cluster_id, function(id) {
      results$initial_results$consensus_results[[as.numeric(id)]]$agreement_score
    })
  )

# Add initial model predictions
for (model in names(results$initial_results$individual_predictions)) {
  results_df[[paste0("initial_", model)]] <- sapply(results_df$cluster_id, function(id) {
    results$initial_results$individual_predictions[[model]][as.numeric(id)]
  })
}

# Reorder columns
results_df <- results_df %>%
  select(
    cluster_id,
    reference_name,
    final_consensus,
    is_controversial,
    everything()
  )

# Save as CSV
output_path <- file.path(project_root,
                        "results/benchmark/reference_comparison/2_evaluation",
                        paste0(tissue, "_results.csv"))
write_csv(results_df, output_path)

# Print preview of the results
print(paste("\nResults preview for", tissue, ":"))
print(results_df)

print(paste("\nResults saved to:", output_path))
