library(tidyverse)

# Set project root
project_root <- "."

# Process HLCA leiden 4 results
tissue <- "HLCA_leiden_4"

# Read the RDS file
results <- readRDS(file.path(project_root, paste0("results/benchmark/popv_comparison/2_evaluation/", tissue, "_results_v2.rds")))

  # Get reference names from the markers RDS file
  markers <- readRDS(file.path(project_root, "results/benchmark/popv_comparison/1_preprocessing/HLCA_leiden_4_markers_v2.rds"))
  reference_data <- names(markers)
  n_clusters <- length(reference_data)



  # Print the number of reference cell types
  print(paste("Number of", tissue, "reference cell types:"))
  print(length(reference_data))

  # Create data frame with initial predictions
  results_df <- tibble(
    cluster_id = as.character(1:n_clusters),
    reference_name = reference_data,
    is_controversial = cluster_id %in% results$controversial_clusters
  ) %>%
    mutate(
      final_consensus = sapply(cluster_id, function(id) {
        if (!is.null(results$final_annotations[[id]])) {
          results$final_annotations[[id]]
        } else {
          NA_character_
        }
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
  write_csv(results_df,
            file.path(project_root, paste0("results/benchmark/popv_comparison/2_evaluation/", tissue, "_results_v2.csv")))

  # Print preview of the results
  print(paste("\n", tissue, "results preview:"))
  print(results_df)
