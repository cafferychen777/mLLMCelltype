# Generic evaluation script for cell type benchmarks with minimal providers
library(LLMCellType)

#' Run consensus evaluation for a given dataset
#' @param dataset_name The name of the dataset
#' @param tissue_name Optional tissue name override
#' @param num_parts Number of parts to split the data into. Default is 2
run_consensus_evaluation <- function(dataset_name, tissue_name = NULL, num_parts = 2) {
  # Set paths
  project_root <- "."

  # Read and parse .env file
  env_lines <- readLines(file.path(project_root, ".env"))
  for(line in env_lines) {
    if(grepl("^[[:alpha:]].*=.*$", line)) {
      parts <- strsplit(line, "=")[[1]]
      key <- trimws(parts[1])
      value <- trimws(gsub('"', '', parts[2]))
      do.call(Sys.setenv, structure(list(value), names = key))
    }
  }

  # Prepare API keys by provider (only anthropic and openai)
  api_keys <- list(
    "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
    "openai" = Sys.getenv("OPENAI_API_KEY")
  )

  # Clean up previous results and cache
  unlink(file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/logs"), recursive = TRUE)
  unlink(file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/cache"), recursive = TRUE)
  unlink(file.path(project_root, sprintf("results/benchmark/reference_comparison/2_evaluation/%s.rds", dataset_name)))

  # Create fresh results directories
  dir.create(file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/logs"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/cache"), recursive = TRUE, showWarnings = FALSE)

  # Load preprocessed data
  markers_list <- readRDS(file.path(project_root, sprintf("results/benchmark/reference_comparison/1_preprocessing/%s_markers.rds", dataset_name)))

  # Print data structure for verification
  print("Loaded data structure:")
  str(markers_list)

  # Print API key status
  print("\nAPI Key Status:")
  print(paste("ANTHROPIC_API_KEY:", if(nchar(Sys.getenv("ANTHROPIC_API_KEY")) > 0) "Set" else "Not Set"))
  print(paste("OPENAI_API_KEY:", if(nchar(Sys.getenv("OPENAI_API_KEY")) > 0) "Set" else "Not Set"))

  # Extract tissue name if not provided
  if (is.null(tissue_name)) {
    if (grepl("^GTEx_DE_", dataset_name)) {
      tissue_name <- gsub("^GTEx_DE_(.+)_top\\d+$", "\\1", dataset_name)
      tissue_name <- gsub("_", " ", tissue_name)
    } else {
      tissue_name <- "human tissue"
    }
  }

  # Split markers_list into parts
  total_clusters <- length(markers_list)
  clusters_per_part <- ceiling(total_clusters / num_parts)
  marker_names <- names(markers_list)
  global_cluster_names <- as.character(seq_len(total_clusters))

  # Process each part
  for (i in 1:num_parts) {
    start_idx <- (i-1) * clusters_per_part + 1
    end_idx <- min(i * clusters_per_part, total_clusters)

    if (start_idx > total_clusters) break

    print(sprintf("\nProcessing Part %d (clusters %d-%d):", i, start_idx, end_idx))
    part_names <- marker_names[start_idx:end_idx]
    part_markers <- markers_list[part_names]

    max_retries <- 7
    retry_count <- 0
    success <- FALSE

    while (!success && retry_count < max_retries) {
      retry_count <- retry_count + 1

      tryCatch({
        # Run consensus annotation with only two models
        results <- interactive_consensus_annotation(
          input = part_markers,
          tissuename = sprintf("%s tissue part %d", tissue_name, i),
          models = c("claude-3-5-sonnet-latest",
                    "gpt-4o"),  # Only using one model from each provider
          api_keys = api_keys,
          log_dir = file.path(project_root, sprintf("results/benchmark/reference_comparison/2_evaluation/logs/part%d", i)),
          cache_dir = file.path(project_root, sprintf("results/benchmark/reference_comparison/2_evaluation/cache/part%d", i))
        )

        saveRDS(results,
                file.path(project_root,
                         sprintf("results/benchmark/reference_comparison/2_evaluation/%s_results_part%d.rds",
                                dataset_name, i)))

        success <- TRUE

      }, error = function(e) {
        message(sprintf("\nAttempt %d failed with error: %s", retry_count, conditionMessage(e)))
        if (retry_count < max_retries) message("Retrying...")
      })
    }

    if (!success) {
      stop(sprintf("Failed to process part %d after %d attempts", i, max_retries))
    }
  }

  # Initialize variables for combining results
  combined_results <- list()
  global_final_annotations <- list()
  global_individual_predictions <- list()
  global_consensus_results <- list()
  global_initial_controversial_clusters <- character(0)
  global_top_controversial_clusters <- character(0)
  global_discussion_logs <- list()

  # Combine results from all parts
  for (i in 1:num_parts) {
    start_idx <- (i-1) * clusters_per_part + 1
    end_idx <- min(i * clusters_per_part, total_clusters)
    part_cluster_names <- global_cluster_names[start_idx:end_idx]

    part_file <- file.path(project_root,
                          sprintf("results/benchmark/reference_comparison/2_evaluation/%s_results_part%d.rds", dataset_name, i))

    if (!file.exists(part_file)) {
      message(sprintf("File %s does not exist, skipping this part.", part_file))
      next
    }

    message(sprintf("Loading and merging part %d ...", i))
    part_results <- readRDS(part_file)

    if(length(part_results$final_annotations) == length(part_cluster_names)) {
      names(part_results$final_annotations) <- part_cluster_names
      global_final_annotations <- c(global_final_annotations, part_results$final_annotations)
    }

    for(model in names(part_results$initial_results$individual_predictions)) {
      preds <- part_results$initial_results$individual_predictions[[model]]
      if(length(preds) == length(part_cluster_names)) {
        names(preds) <- part_cluster_names
      }
      if(is.null(global_individual_predictions[[model]])) {
        global_individual_predictions[[model]] <- preds
      } else {
        global_individual_predictions[[model]] <- c(global_individual_predictions[[model]], preds)
      }
    }

    consensus <- part_results$initial_results$consensus_results
    if(length(consensus) == length(part_cluster_names)) {
      names(consensus) <- part_cluster_names
      global_consensus_results <- c(global_consensus_results, consensus)
    }

    local_controversial <- part_results$initial_results$controversial_clusters
    if(length(local_controversial) > 0) {
      local_idx <- as.numeric(local_controversial)
      local_global_names <- part_cluster_names[local_idx]
      global_initial_controversial_clusters <- c(global_initial_controversial_clusters, local_global_names)
    }

    local_top <- part_results$controversial_clusters
    if(length(local_top) > 0) {
      local_idx <- as.numeric(local_top)
      local_global_names <- part_cluster_names[local_idx]
      global_top_controversial_clusters <- c(global_top_controversial_clusters, local_global_names)
    }

    if(length(part_results$discussion_logs) > 0) {
      for(local_idx in names(part_results$discussion_logs)) {
        idx <- as.numeric(local_idx)
        global_name <- part_cluster_names[idx]
        global_discussion_logs[[global_name]] <- part_results$discussion_logs[[local_idx]]
      }
    }
  }

  global_initial_controversial_clusters <- unique(global_initial_controversial_clusters)
  global_top_controversial_clusters <- unique(global_top_controversial_clusters)

  combined_results$final_annotations <- global_final_annotations
  combined_results$initial_results$individual_predictions <- global_individual_predictions
  combined_results$initial_results$consensus_results <- global_consensus_results
  combined_results$initial_results$controversial_clusters <- global_initial_controversial_clusters
  combined_results$controversial_clusters <- global_top_controversial_clusters
  combined_results$discussion_logs <- global_discussion_logs

  saveRDS(combined_results,
          file.path(project_root,
                    sprintf("results/benchmark/reference_comparison/2_evaluation/%s_results.rds", dataset_name)))

  print(sprintf("\nEvaluation complete for %s. Combined results saved to %s_results.rds", dataset_name, dataset_name))
}

# Run with minimal providers
run_consensus_evaluation("Sun_2020_top20", tissue_name = "While this study was designed to investigate age-related differences in COVID-19 susceptibility, the dataset we are using for annotation contains only normal (non-COVID) lung tissue samples. The researchers conducted snRNA-seq analysis on 46,500 nuclei from healthy human lung samples, collected from donors of three distinct age groups: approximately 30 weeks gestational age (premature births), 3 years old, and 30 years old. The study included three normal lung samples per age group, with a balanced representation of both males and females. Among the nine healthy donors, five were Caucasian, one was African American, and three had undocumented ancestry. All samples were obtained from flash-frozen biopsies of equivalent small airway regions of the right middle lobe from normal lung tissue. For one of the 3-year-old donors (D032), technical replicates were generated. The cell type annotations in this dataset follow a five-level hierarchy: Level 1 represents basic cell lineages (Immune, Epithelial, Stroma, Endothelial) reflecting developmental origin; Level 2 covers tissue-specific classifications (e.g., Alveolar epithelium, Myeloid, Fibroblast lineage); Level 3 describes specific cell types (e.g., AT2, Fibroblasts, Macrophages) reflecting main functional identity; Level 4 defines detailed subtypes and states (e.g., Alveolar macrophages, Interstitial macrophages); and Level 5 represents the most granular subtypes with specific molecular features. For this analysis, please focus on Level 3 annotations, which requires identifying specific cell types with their main functional characteristics in normal lung tissue.")
