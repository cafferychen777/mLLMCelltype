# Evaluation script for HLCA leiden 2 cell type benchmark
library(LLMCellType)

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

# Clean up previous results and cache
unlink(file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/logs"), recursive = TRUE)
unlink(file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/cache"), recursive = TRUE)
unlink(file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_3_results.rds"))

# Create fresh results directories
dir.create(file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/logs"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/cache"), recursive = TRUE, showWarnings = FALSE)

# Load preprocessed data
markers_list <- readRDS(file.path(project_root, "results/benchmark/popv_comparison/1_preprocessing/HLCA_leiden_3_markers.rds"))

# Print data structure for verification
print("Loaded data structure:")
str(markers_list)

# Print API key status
print("\nAPI Key Status:")
print(paste("ANTHROPIC_API_KEY:", if(nchar(Sys.getenv("ANTHROPIC_API_KEY")) > 0) "Set" else "Not Set"))
print(paste("GEMINI_API_KEY:", if(nchar(Sys.getenv("GEMINI_API_KEY")) > 0) "Set" else "Not Set"))
print(paste("QWEN_API_KEY:", if(nchar(Sys.getenv("QWEN_API_KEY")) > 0) "Set" else "Not Set"))
print(paste("OPENAI_API_KEY:", if(nchar(Sys.getenv("OPENAI_API_KEY")) > 0) "Set" else "Not Set"))

# Split markers_list into parts
total_clusters <- length(markers_list)
num_parts <- 8
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

  # Run consensus annotation for this part with retry logic
  max_retries <- 3
  retry_count <- 0
  success <- FALSE

  while (!success && retry_count < max_retries) {
    tryCatch({
      results <- interactive_consensus_annotation(
        input = part_markers,
        tissuename = sprintf("Background of Human Lung Cell Atlas (HLCA):\n\nHLCA is a comprehensive integration of 49 datasets of the human respiratory system, combining over 2.4 million cells from 486 individuals. This atlas provides consensus cell type annotations with matching marker genes, including rare and previously undescribed cell types. HLCA captures demographic variations (age, sex, BMI) and spatial gene expression patterns along the bronchial tree. It serves as a reference for disease studies, revealing shared cell states across conditions like COVID-19, pulmonary fibrosis, and lung carcinoma.\n\nAnnotation Hierarchy:\nLevel 1: Basic cell lineages (e.g., immune cells, epithelial cells, endothelial cells, stromal cells)\nLevel 2: Major cell lineage classifications\nLevel 3: Specific cell type lineages\nLevel 4: Defined cell types\nLevel 5: Cell subtypes\n\nThis is part %d of the analysis. We are currently working on Level 3 annotations - Specific cell type lineages. Please provide cell type annotations at Level 3, focusing on specific cell type lineages rather than broader classifications. For example, if you see markers indicating T cells, annotate as 'CD4+ T cells' or 'CD8+ T cells' rather than just broadly as 'T cells' or more specifically as 'naive CD8+ T cells' or 'memory CD4+ T cells'.", i),
        models = c("claude-3-5-sonnet-latest",
                   "claude-3-5-haiku-latest",
                   "gemini-1.5-pro",
                   "gemini-2.0-flash-exp",
                   "qwen-max-2025-01-25",
                   "gpt-4o"),
        api_keys = list(
          "claude-3-5-sonnet-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
          "claude-3-5-haiku-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
          "gemini-1.5-pro" = Sys.getenv("GEMINI_API_KEY"),
          "gemini-2.0-flash-exp" = Sys.getenv("GEMINI_API_KEY"),
          "qwen-max-2025-01-25" = Sys.getenv("QWEN_API_KEY"),
          "gpt-4o" = Sys.getenv("OPENAI_API_KEY")
        ),
        log_dir = file.path(project_root, sprintf("results/benchmark/popv_comparison/2_evaluation/logs/part%d", i)),
        cache_dir = file.path(project_root, sprintf("results/benchmark/popv_comparison/2_evaluation/cache/part%d", i))
      )

      # If we get here, the operation was successful
      success <- TRUE

      # Save results for this part
      saveRDS(results,
              file.path(project_root,
                        sprintf("results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_3_results_part%d.rds", i)))

      print(sprintf("Part %d evaluation complete. Results saved to HLCA_leiden_3_results_part%d.rds", i, i))

    }, error = function(e) {
      retry_count <<- retry_count + 1
      if (retry_count < max_retries) {
        print(sprintf("Error in part %d: %s\nRetrying attempt %d of %d after a delay...", i, e$message, retry_count + 1, max_retries))
        # Wait for a bit before retrying (increasing delay for each retry)
        Sys.sleep(retry_count * 60)  # Wait 1 minute on first retry, 2 on second, etc.
      } else {
        print(sprintf("Failed to complete part %d after %d attempts. Error: %s", i, max_retries, e$message))
      }
    })
  }

  if (!success) {
    print(sprintf("Skipping part %d due to repeated failures", i))
    next
  }
}

# Temporary variables for global merging
global_final_annotations <- list()
global_individual_predictions <- list()  # Stored by model
global_consensus_results <- list()
global_discussion_logs <- list()
global_initial_controversial_clusters <- character(0)
global_top_controversial_clusters <- character(0)

combined_results <- list(
  initial_results = list(
    individual_predictions = list(),
    consensus_results = list(),
    controversial_clusters = character(0)
  ),
  final_annotations = list(),
  controversial_clusters = character(0),
  discussion_logs = list(),
  session_id = format(Sys.time(), "%Y%m%d_%H%M%S")
)

# Loop through each part
for(i in 1:num_parts) {
  # Calculate global cluster names corresponding to this part
  start_idx <- (i - 1) * clusters_per_part + 1
  end_idx <- min(i * clusters_per_part, total_clusters)
  part_cluster_names <- global_cluster_names[start_idx:end_idx]

  part_file <- file.path(project_root,
                         sprintf("results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_3_results_part%d.rds", i))
  if (!file.exists(part_file)) {
    message(sprintf("File %s does not exist, skipping this part.", part_file))
    next
  }
  message(sprintf("Loading and merging part %d ...", i))
  part_results <- readRDS(part_file)

  ### 1. Merge final_annotations ###
  # Assume part_results$final_annotations order corresponds one-to-one with part_cluster_names
  if(length(part_results$final_annotations) == length(part_cluster_names)) {
    names(part_results$final_annotations) <- part_cluster_names
    global_final_annotations <- c(global_final_annotations, part_results$final_annotations)
  } else {
    message(sprintf("Warning: part %d final_annotations length (%d) does not match expected length (%d)!",
                    i, length(part_results$final_annotations), length(part_cluster_names)))
  }

  ### 2. Merge individual_predictions ###
  # part_results$initial_results$individual_predictions is a list, each element is a prediction list for this part by a model
  for(model in names(part_results$initial_results$individual_predictions)) {
    preds <- part_results$initial_results$individual_predictions[[model]]
    # If prediction count matches this part's clusters count, assign names
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
  # part_results$initial_results$consensus_results is also ordered sequentially
  consensus <- part_results$initial_results$consensus_results
  if(length(consensus) == length(part_cluster_names)) {
    names(consensus) <- part_cluster_names
    global_consensus_results <- c(global_consensus_results, consensus)
  }

  ### 4. Merge controversial_clusters (initial stage) ###
  # part_results$initial_results$controversial_clusters may store indices (character type) within this part
  local_controversial <- part_results$initial_results$controversial_clusters
  if(length(local_controversial) > 0) {
    local_idx <- as.numeric(local_controversial)
    # For this part, these numeric indices correspond to global cluster names
    local_global_names <- part_cluster_names[local_idx]
    global_initial_controversial_clusters <- c(global_initial_controversial_clusters, local_global_names)
  }

  ### 5. Merge controversial_clusters (top level) ###
  # part_results$controversial_clusters at top level also stores indices (character type) of controversial clusters
  local_top <- part_results$controversial_clusters
  if(length(local_top) > 0) {
    local_idx <- as.numeric(local_top)
    local_global_names <- part_cluster_names[local_idx]
    global_top_controversial_clusters <- c(global_top_controversial_clusters, local_global_names)
  }

  ### 6. Merge discussion_logs ###
  # part_results$discussion_logs keys are usually indices (character type) within this part
  if(length(part_results$discussion_logs) > 0) {
    for(local_idx in names(part_results$discussion_logs)) {
      # Convert to numeric index, then map to global cluster name
      idx <- as.numeric(local_idx)
      global_name <- part_cluster_names[idx]
      global_discussion_logs[[global_name]] <- part_results$discussion_logs[[local_idx]]
    }
  }
}

# To avoid duplicates, take unique values
global_initial_controversial_clusters <- unique(global_initial_controversial_clusters)
global_top_controversial_clusters <- unique(global_top_controversial_clusters)

# Assign to combined_results
combined_results$final_annotations <- global_final_annotations
combined_results$initial_results$individual_predictions <- global_individual_predictions
combined_results$initial_results$consensus_results <- global_consensus_results
combined_results$initial_results$controversial_clusters <- global_initial_controversial_clusters
combined_results$controversial_clusters <- global_top_controversial_clusters
combined_results$discussion_logs <- global_discussion_logs

# Save merged results
combined_file <- file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_3_results.rds")
saveRDS(combined_results, combined_file)