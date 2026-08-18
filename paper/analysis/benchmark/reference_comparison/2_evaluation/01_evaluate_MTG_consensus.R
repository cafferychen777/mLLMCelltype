# Generic evaluation script for cell type benchmarks
library(mLLMCelltype)

#' Run consensus evaluation for a given dataset
#' @param dataset_name The name of the dataset (e.g., "GTEx_DE_breast_top30", "GTEx_DE_skin_top30", "HCL")
#' @param tissue_name Optional tissue name override. If NULL, will be extracted from dataset_name
#' @param num_parts Number of parts to split the data into. Default is 2
run_consensus_evaluation <- function(dataset_name, tissue_name = NULL, num_parts = 1) {
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

  # Prepare API keys by provider
  api_keys <- list(
    "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
    "gemini" = Sys.getenv("GEMINI_API_KEY"),
    "qwen" = Sys.getenv("QWEN_API_KEY"),
    "openai" = Sys.getenv("OPENAI_API_KEY"),
    "zhipu" = Sys.getenv("ZHIPU_API_KEY"),
    "stepfun" = Sys.getenv("STEPFUN_API_KEY"),
    "minimax" = Sys.getenv("MINIMAX_API_KEY")
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
  print(paste("GEMINI_API_KEY:", if(nchar(Sys.getenv("GEMINI_API_KEY")) > 0) "Set" else "Not Set"))
  print(paste("QWEN_API_KEY:", if(nchar(Sys.getenv("QWEN_API_KEY")) > 0) "Set" else "Not Set"))
  print(paste("OPENAI_API_KEY:", if(nchar(Sys.getenv("OPENAI_API_KEY")) > 0) "Set" else "Not Set"))
  print(paste("ZHIPU_API_KEY:", if(nchar(Sys.getenv("ZHIPU_API_KEY")) > 0) "Set" else "Not Set"))
  print(paste("STEPFUN_API_KEY:", if(nchar(Sys.getenv("STEPFUN_API_KEY")) > 0) "Set" else "Not Set"))
  print(paste("MINIMAX_API_KEY:", if(nchar(Sys.getenv("MINIMAX_API_KEY")) > 0) "Set" else "Not Set"))

  # Extract tissue name if not provided
  if (is.null(tissue_name)) {
    if (grepl("^GTEx_DE_", dataset_name)) {
      # Extract tissue name from GTEx dataset name
      tissue_name <- gsub("^GTEx_DE_(.+)_top\\d+$", "\\1", dataset_name)
      tissue_name <- gsub("_", " ", tissue_name)
    } else {
      # For other datasets like HCL, use generic name
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
    # Calculate start and end indices for this part
    start_idx <- (i-1) * clusters_per_part + 1
    end_idx <- min(i * clusters_per_part, total_clusters)

    # Skip if we've processed all clusters
    if (start_idx > total_clusters) break

    # Get subset of markers for this part
    print(sprintf("\nProcessing Part %d (clusters %d-%d):", i, start_idx, end_idx))
    part_names <- marker_names[start_idx:end_idx]
    part_markers <- markers_list[part_names]

    # Add retry logic for consensus annotation
    max_retries <- 2
    retry_count <- 0
    success <- FALSE

    while (!success && retry_count < max_retries) {
      retry_count <- retry_count + 1

      tryCatch({
        # Run consensus annotation for this part
        results <- interactive_consensus_annotation(
          input = part_markers,
          tissuename = sprintf("%s tissue part %d", tissue_name, i),
          models = c("claude-3-5-sonnet-latest",
                    "claude-3-5-haiku-latest",
                    "gemini-1.5-pro",
                    "gemini-2.0-flash-exp",
                    "qwen-max-2025-01-25",
                    "gpt-4o",
                    "step-2-16k",
                    "glm-4-plus",
                    "minimax-text-01"),
          api_keys = api_keys,  # 添加 api_keys 参数
          log_dir = file.path(project_root, sprintf("results/benchmark/reference_comparison/2_evaluation/logs/part%d", i)),
          cache_dir = file.path(project_root, sprintf("results/benchmark/reference_comparison/2_evaluation/cache/part%d", i))
        )

        # Save part results
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
    # Calculate cluster names for this part
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

    ### 1. Merge final_annotations ###
    if(length(part_results$final_annotations) == length(part_cluster_names)) {
      names(part_results$final_annotations) <- part_cluster_names
      global_final_annotations <- c(global_final_annotations, part_results$final_annotations)
    } else {
      message(sprintf("Warning: part %d final_annotations length (%d) does not match expected length (%d)!",
                     i, length(part_results$final_annotations), length(part_cluster_names)))
    }

    ### 2. Merge individual_predictions ###
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

    ### 3. Merge consensus_results ###
    consensus <- part_results$initial_results$consensus_results
    if(length(consensus) == length(part_cluster_names)) {
      names(consensus) <- part_cluster_names
      global_consensus_results <- c(global_consensus_results, consensus)
    }

    ### 4. Merge controversial_clusters (initial stage) ###
    local_controversial <- part_results$initial_results$controversial_clusters
    if(length(local_controversial) > 0) {
      local_idx <- as.numeric(local_controversial)
      local_global_names <- part_cluster_names[local_idx]
      global_initial_controversial_clusters <- c(global_initial_controversial_clusters, local_global_names)
    }

    ### 5. Merge controversial_clusters (top level) ###
    local_top <- part_results$controversial_clusters
    if(length(local_top) > 0) {
      local_idx <- as.numeric(local_top)
      local_global_names <- part_cluster_names[local_idx]
      global_top_controversial_clusters <- c(global_top_controversial_clusters, local_global_names)
    }

    ### 6. Merge discussion_logs ###
    if(length(part_results$discussion_logs) > 0) {
      for(local_idx in names(part_results$discussion_logs)) {
        idx <- as.numeric(local_idx)
        global_name <- part_cluster_names[idx]
        global_discussion_logs[[global_name]] <- part_results$discussion_logs[[local_idx]]
      }
    }
  }

  # Remove duplicates from controversial clusters
  global_initial_controversial_clusters <- unique(global_initial_controversial_clusters)
  global_top_controversial_clusters <- unique(global_top_controversial_clusters)

  # Assemble final combined results
  combined_results$final_annotations <- global_final_annotations
  combined_results$initial_results$individual_predictions <- global_individual_predictions
  combined_results$initial_results$consensus_results <- global_consensus_results
  combined_results$initial_results$controversial_clusters <- global_initial_controversial_clusters
  combined_results$controversial_clusters <- global_top_controversial_clusters
  combined_results$discussion_logs <- global_discussion_logs

  # Save combined results
  saveRDS(combined_results,
          file.path(project_root,
                    sprintf("results/benchmark/reference_comparison/2_evaluation/%s_results.rds", dataset_name)))

  print(sprintf("\nEvaluation complete for %s. Combined results saved to %s_results.rds", dataset_name, dataset_name))
}

run_consensus_evaluation("MTG", tissue_name = "Dissection: Cerebral cortex (Cx) - Middle Temporal Gyrus - MTG")
