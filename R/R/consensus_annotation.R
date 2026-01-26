# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Get initial predictions from all models
#'
#' This function retrieves initial cell type predictions from all specified models.
#' It is an internal helper function used by the interactive_consensus_annotation function.
#'
#' @keywords internal
get_initial_predictions <- function(input, tissue_name, models, api_keys, top_gene_count, base_urls = NULL) {
  log_info("Phase 1: Getting initial predictions from all models...", list(
    models_count = length(models),
    models = models
  ))
  message("\nPhase 1: Getting initial predictions from all models...")

  # Initialize tracking variables
  individual_predictions <- list()
  successful_models <- character(0)

  # Get predictions from each model
  for (model in models) {
    api_key <- get_api_key(model, api_keys)

    if (is.null(api_key)) {
      warning_msg <- sprintf("No API key found for model '%s' (provider: %s). This model will be skipped.",
                            model, get_provider(model))
      warning(warning_msg)
      log_warn(warning_msg, list(model = model, provider = get_provider(model)))
      next
    }

    tryCatch({
      predictions <- annotate_cell_types(
        input = input,
        tissue_name = tissue_name,
        model = model,
        api_key = api_key,
        top_gene_count = top_gene_count,
        base_urls = base_urls
      )
      individual_predictions[[model]] <- predictions
      successful_models <- c(successful_models, model)
    }, error = function(e) {
      warning_msg <- sprintf("Failed to get predictions from %s: %s", model, e$message)
      warning(warning_msg)
      log_warn(warning_msg, list(model = model, error = e$message))
    })
  }

  if (length(successful_models) == 0) {
    stop("No models successfully completed predictions. Please check API keys and model availability.")
  }

  return(list(
    individual_predictions = individual_predictions,
    successful_models = successful_models
  ))
}

#' Identify controversial clusters based on consensus analysis
#'
#
#
#
#
#
#' @keywords internal
identify_controversial_clusters <- function(input, individual_predictions, controversy_threshold, entropy_threshold, api_keys, consensus_check_model = NULL) {
  # For each cluster, check consensus
  clusters <- if (inherits(input, 'list')) {
    names(input)
  } else {
    unique(input$cluster)
  }
  
  log_info("Phase 2: Identifying controversial clusters...", list(
    clusters_count = length(clusters),
    entropy_threshold = entropy_threshold,
    controversy_threshold = controversy_threshold
  ))
  message("\nPhase 2: Identifying controversial clusters...")

  # Initialize consensus tracking
  consensus_results <- list()
  controversial_clusters <- character(0)
  final_annotations <- list()

  # Restructure individual_predictions to be indexed by cluster_id
  # This handles the case where individual_predictions are returned as text lines from models
  structured_predictions <- list()

  for (model_name in names(individual_predictions)) {
    model_preds <- individual_predictions[[model_name]]

    # Check if model_preds is already structured by cluster_id
    if (is.list(model_preds) && !is.null(names(model_preds))) {
      structured_predictions[[model_name]] <- model_preds
    } else if (is.character(model_preds)) {
      # Parse text lines into a structured format
      model_structured <- list()

      # Process each line which should be in format: "cluster_id: cell_type"
      for (line in model_preds) {
        # Skip empty lines
        if (trimws(line) == "") next

        # Try to parse the line as "cluster_id: cell_type"
        parts <- strsplit(line, ":", fixed = TRUE)[[1]]
        if (length(parts) >= 2) {
          cluster_num <- trimws(parts[1])
          cell_type <- trimws(paste(parts[-1], collapse = ":"))
          model_structured[[cluster_num]] <- cell_type
        } else {
          # Try to parse other formats, such as formats with numeric indices like "1. cell_type"
          # Match numeric index formats, such as "1. ", "1- ", "1 "
          number_match <- regexpr("^\\s*\\d+[\\.-]?\\s+", line)
          if (number_match > 0) {
            # Extract the index part
            number_part <- substr(line, 1, attr(number_match, "match.length"))
            # Extract the number
            number <- as.numeric(gsub("[^0-9]", "", number_part))
            # Extract the cell type part
            cell_type <- trimws(substr(line, attr(number_match, "match.length") + 1, nchar(line)))
            # Numeric indices usually start from 1, while cluster IDs usually start from 0, so conversion is needed
            cluster_num <- as.character(number - 1)
            model_structured[[cluster_num]] <- cell_type
          }
        }
      }

      # If no predictions were found in the above processing, try using index position
      # Get all cluster IDs
      all_clusters <- if (inherits(input, 'list')) {
        names(input)
      } else {
        as.character(unique(input$cluster))
      }

      # For each cluster ID, if no prediction is found, try using index position
      for (cluster_id in all_clusters) {
        # Try to convert cluster_id to numeric safely
        cluster_idx <- suppressWarnings(as.numeric(cluster_id))
        if (is.null(model_structured[[cluster_id]]) && !is.na(cluster_idx) && length(model_preds) > cluster_idx) {
          # Assume predictions are arranged in order of cluster ID
          index <- cluster_idx + 1  # Convert from 0-based to 1-based
          if (index <= length(model_preds)) {
            potential_cell_type <- trimws(model_preds[index])
            # Check if it contains ":", if so, extract the part after it
            if (grepl(":", potential_cell_type, fixed = TRUE)) {
              parts <- strsplit(potential_cell_type, ":", fixed = TRUE)[[1]]
              if (length(parts) >= 2) {
                model_structured[[cluster_id]] <- trimws(paste(parts[-1], collapse = ":"))
              }
            } else {
              # Check if it contains a numeric index format
              number_match <- regexpr("^\\s*\\d+[\\.-]?\\s+", potential_cell_type)
              if (number_match > 0) {
                # Extract the cell type part
                model_structured[[cluster_id]] <- trimws(substr(potential_cell_type, attr(number_match, "match.length") + 1, nchar(potential_cell_type)))
              } else {
                # Does not contain ":" or numeric index, use directly
                model_structured[[cluster_id]] <- potential_cell_type
              }
            }
          }
        }
      }

      structured_predictions[[model_name]] <- model_structured
    }
  }

  for (cluster_id in clusters) {
    # Use original cluster ID for log output, no conversion needed
    log_info(sprintf("Analyzing cluster %s...", cluster_id), list(cluster_id = cluster_id))
    message(sprintf("\nAnalyzing cluster %s...", cluster_id))

    # Get predictions for this cluster from each model
    cluster_predictions <- sapply(structured_predictions, function(x) {
      if (is.null(x[[as.character(cluster_id)]])) NA else x[[as.character(cluster_id)]]
    })
    valid_predictions <- cluster_predictions[!is.na(cluster_predictions)]

    if (length(valid_predictions) == 0) {
      log_warn(sprintf("No valid predictions for cluster %s. Marking as controversial.", cluster_id), list(cluster_id = cluster_id))
      message(sprintf("No valid predictions for cluster %s. Marking as controversial.", cluster_id))
      controversial_clusters <- c(controversial_clusters, as.character(cluster_id))
      next
    }

    # Calculate agreement score
    # Parameters are passed to check_consensus and used in prompt template to instruct LLM # nolint
    initial_consensus <- check_consensus(valid_predictions, api_keys, controversy_threshold, entropy_threshold, consensus_check_model)
    consensus_results[[as.character(cluster_id)]] <- initial_consensus

    # If no consensus is reached or the consensus metrics indicate high uncertainty, mark it as controversial.
    # Use both consensus proportion and entropy for decision making
    if (!initial_consensus$reached ||
        initial_consensus$consensus_proportion < controversy_threshold ||
        initial_consensus$entropy > entropy_threshold) {

      log_info(sprintf("Cluster %s marked as controversial", cluster_id), list(
        cluster_id = cluster_id,
        reached_consensus = initial_consensus$reached,
        consensus_proportion = initial_consensus$consensus_proportion,
        entropy = initial_consensus$entropy
      ))

      message(sprintf("Cluster %s marked as controversial (reached: %s, consensus proportion: %.2f, entropy: %.2f)",
                     cluster_id, initial_consensus$reached,
                     initial_consensus$consensus_proportion, initial_consensus$entropy))

      controversial_clusters <- c(controversial_clusters, as.character(cluster_id))
    } else {
      # Process non-controversial clusters
      final_annotations[[as.character(cluster_id)]] <- select_best_prediction(initial_consensus, valid_predictions)

      log_info(sprintf("Consensus reached for cluster %s", cluster_id), list(
        cluster_id = cluster_id,
        consensus_proportion = initial_consensus$consensus_proportion,
        entropy = initial_consensus$entropy,
        selected_cell_type = final_annotations[[as.character(cluster_id)]]
      ))

      message(sprintf("Consensus reached for cluster %s (consensus proportion: %.2f, entropy: %.2f, selected: %s)",
                     cluster_id, initial_consensus$consensus_proportion,
                     initial_consensus$entropy, final_annotations[[as.character(cluster_id)]]))
    }
  }

  return(list(
    consensus_results = consensus_results,
    controversial_clusters = controversial_clusters,
    final_annotations = final_annotations
  ))
}

#' Select the best prediction from consensus results
#'
#
#
#
#' @keywords internal
select_best_prediction <- function(consensus_result, valid_predictions) {
  # If we have a majority prediction from Claude, use it
  if (!is.na(consensus_result$majority_prediction) &&
      !is.null(consensus_result$majority_prediction) &&
      is.character(consensus_result$majority_prediction) &&
      consensus_result$majority_prediction != "Unknown" &&
      consensus_result$majority_prediction != "") {
    return(consensus_result$majority_prediction)
  }

  # Fallback to frequency-based approach if Claude didn't provide a valid majority prediction
  # Calculate the frequency of occurrence for each prediction
  prediction_counts <- table(valid_predictions)
  # Find the prediction with the highest frequency of occurrence
  max_count <- max(prediction_counts)
  most_common_predictions <- names(prediction_counts[prediction_counts == max_count])

  if (length(most_common_predictions) == 1) {
    # If there is only one most common prediction, use it directly.
    return(most_common_predictions[1])
  } else {
    # If there are multiple most common predictions, use the longest (most detailed) one.
    return(most_common_predictions[which.max(nchar(most_common_predictions))])
  }
}

#' Process controversial clusters through discussion
#'
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#' @keywords internal
process_controversial_clusters <- function(controversial_clusters, input, tissue_name,
                                          successful_models, api_keys, individual_predictions,
                                          top_gene_count, controversy_threshold, entropy_threshold, max_discussion_rounds,
                                          cache_manager, use_cache, consensus_check_model = NULL, force_rerun = FALSE) {

  if (length(controversial_clusters) == 0) {
    log_info("No controversial clusters found. All clusters have reached consensus.")
    message("\nNo controversial clusters found. All clusters have reached consensus.")
    return(list(
      discussion_logs = list(),
      final_annotations = list()
    ))
  }

  log_info(sprintf("Phase 3: Starting discussions for %d controversial clusters...",
                   length(controversial_clusters)), list(
    controversial_count = length(controversial_clusters),
    clusters = controversial_clusters
  ))
  message(sprintf("\nPhase 3: Starting discussions for %d controversial clusters...",
                 length(controversial_clusters)))

  discussion_logs <- list()
  final_annotations <- list()

  for (cluster_id in controversial_clusters) {
    # Ensure cluster_id is a string type
    char_cluster_id <- as.character(cluster_id)
    log_info(sprintf("Starting discussion for cluster %s...", char_cluster_id), list(
      cluster_id = char_cluster_id
    ))
    message(sprintf("\nStarting discussion for cluster %s...", char_cluster_id))

    # Check cache
    cached_result <- NULL
    if (use_cache && !force_rerun) {
      cache_key <- cache_manager$generate_key(input, successful_models, char_cluster_id)
      cache_debug <- Sys.getenv("LLMCELLTYPE_DEBUG_CACHE") == "TRUE"

      if (cache_debug) {
        message(sprintf("[DEBUG] Cache check for cluster %s: ", char_cluster_id))
      }

      has_cache <- cache_manager$has_cache(cache_key)

      if (cache_debug) {
        message(sprintf("has_cache = %s", has_cache))
      }

      if (has_cache) {
        # Use cached results
        log_info(sprintf("Loading cached result for cluster %s", char_cluster_id), list(
          cluster_id = char_cluster_id,
          cache_key = cache_key
        ))
        message(sprintf("Loading cached result for cluster %s", char_cluster_id))

        cached_result <- cache_manager$load_from_cache(cache_key)

        if (cache_debug) {
          message(sprintf("[INFO] Successfully loaded cached result for cluster %s", cluster_id))
        }
      }
    } else if (force_rerun) {
      log_info(sprintf("Force rerun enabled, skipping cache for cluster %s", char_cluster_id))
    }

    # Use cached results or perform discussion
    if (!is.null(cached_result)) {
      # Use cached results
      discussion_result <- cached_result$discussion_log
      final_annotation <- cached_result$annotation

      log_info(sprintf("Using cached result for cluster %s", char_cluster_id), list(
        cluster_id = char_cluster_id
      ))
      message(sprintf("Using cached result for cluster %s", char_cluster_id))
    } else {
      # Perform discussion
      # Parameters are passed through to check_consensus and used in prompt template to instruct LLM # nolint
      discussion_result <- facilitate_cluster_discussion(
        cluster_id = char_cluster_id,
        input = input,
        tissue_name = tissue_name,
        models = successful_models,  # Only use models that worked in initial phase
        api_keys = api_keys,
        initial_predictions = individual_predictions,
        top_gene_count = top_gene_count,
        max_rounds = max_discussion_rounds,
        controversy_threshold = controversy_threshold,
        entropy_threshold = entropy_threshold,
        consensus_check_model = consensus_check_model
      )

      # Get results from the last round of discussion
      last_round_index <- length(discussion_result$rounds)
      last_round <- discussion_result$rounds[[last_round_index]]

      # Extract and clean majority_prediction
      final_annotation <- clean_annotation(last_round$consensus_result$majority_prediction)

      # Save to cache - fix cache content structure
      if (use_cache) {
        cache_key <- cache_manager$generate_key(input, successful_models, char_cluster_id)
        cache_data <- list(
          annotation = final_annotation,  # Use the correct final_annotation variable
          discussion_log = discussion_result,
          is_controversial = TRUE
        )
        cache_manager$save_to_cache(cache_key, cache_data)
        log_info(sprintf("Saved result to cache for cluster %s", char_cluster_id), list(
          cluster_id = char_cluster_id
        ))
      }
    }

    # Ensure cluster_id in discussion_result is a string type
    if (!is.null(discussion_result) && !is.character(discussion_result$cluster_id)) {
      discussion_result$cluster_id <- char_cluster_id
    }

    discussion_logs[[char_cluster_id]] <- discussion_result
    final_annotations[[char_cluster_id]] <- final_annotation

    log_info(sprintf("Completed discussion for cluster %s", char_cluster_id), list(
      cluster_id = char_cluster_id
    ))
    message(sprintf("Completed discussion for cluster %s", char_cluster_id))
  }

  return(list(
    discussion_logs = discussion_logs,
    final_annotations = final_annotations
  ))
}

#' Clean annotation text by removing prefixes and extra whitespace
#'
#
#
#' @keywords internal
clean_annotation <- function(annotation) {
  if (is.null(annotation) || is.na(annotation)) {
    return("Annotation_Missing")
  }

  # Remove numbered prefixes like "1. ", "1: ", "1- ", etc.
  annotation <- gsub("^\\d+[\\.:\\-\\s]+\\s*", "", annotation)
  # Remove "CELL TYPE:" prefix
  annotation <- gsub("^CELL\\s*TYPE[\\s:]*", "", annotation)
  # Final trim of whitespace
  annotation <- trimws(annotation)

  return(annotation)
}

#' Combine results from all phases of consensus annotation
#'
#
#
#
#
#' @keywords internal
combine_results <- function(initial_results, controversy_results, discussion_results) {
  # Combine final annotations from non-controversial and controversial clusters
  final_annotations <- controversy_results$final_annotations

  # Create a mapping table for discussion results
  discussion_results_map <- list()
  for (cluster_id in names(discussion_results$discussion_logs)) {
    # Ensure cluster_id is a string type
    char_cluster_id <- as.character(cluster_id)
    # Get discussion log
    log <- discussion_results$discussion_logs[[char_cluster_id]]

    # Check discussion rounds
    if (length(log$rounds) > 0) {
      last_round <- log$rounds[[length(log$rounds)]]

      # Check if there is a consensus result
      if ("consensus_result" %in% names(last_round)) {
        majority_prediction <- last_round$consensus_result$majority_prediction
        discussion_results_map[[char_cluster_id]] <- majority_prediction
      }
    }
  }

  # Update final annotations using the mapping table
  for (cluster_id in names(discussion_results_map)) {
    # Ensure cluster_id is a string type
    char_cluster_id <- as.character(cluster_id)
    final_annotations[[char_cluster_id]] <- discussion_results_map[[char_cluster_id]]
  }

  # Verify consistency between final annotations and discussion results
  for (cluster_id in names(discussion_results_map)) {
    # Ensure cluster_id is a string type
    char_cluster_id <- as.character(cluster_id)
    if (char_cluster_id %in% names(final_annotations)) {
      # First handle NA values with descriptive replacements
      if (is.na(final_annotations[[char_cluster_id]])) {
        log_warn(sprintf("Cluster %s final annotation is NA, replacing with descriptive placeholder", char_cluster_id), list(
          cluster_id = char_cluster_id
        ))
        final_annotations[[char_cluster_id]] <- "Data_Missing_In_Final_Annotations"
      }
      if (is.na(discussion_results_map[[char_cluster_id]])) {
        log_warn(sprintf("Cluster %s discussion result is NA, replacing with descriptive placeholder", char_cluster_id), list(
          cluster_id = char_cluster_id
        ))
        discussion_results_map[[char_cluster_id]] <- "Data_Missing_In_Discussion_Results"
      }

      # Now we can safely compare without NA concerns
      if (final_annotations[[char_cluster_id]] != discussion_results_map[[char_cluster_id]]) {
        log_warn(sprintf("Cluster %s final annotation differs from discussion result, corrected", char_cluster_id), list(
          cluster_id = char_cluster_id,
          final_annotation = final_annotations[[char_cluster_id]],
          discussion_result = discussion_results_map[[char_cluster_id]]
        ))
        final_annotations[[char_cluster_id]] <- discussion_results_map[[char_cluster_id]]
      }
    }
  }

  # Check consistency with initial predictions
  for (cluster_id in controversy_results$controversial_clusters) {
    # Ensure cluster_id is a string type
    char_cluster_id <- as.character(cluster_id)
    # Collect all initial predictions
    initial_predictions <- list()
    for (model in names(initial_results$individual_predictions)) {
      if (char_cluster_id %in% names(initial_results$individual_predictions[[model]])) {
        prediction <- initial_results$individual_predictions[[model]][[char_cluster_id]]
        initial_predictions[[model]] <- prediction
      }
    }

    # Check if initial predictions are consistent
    unique_predictions <- unique(unlist(initial_predictions))
    if (length(unique_predictions) == 1 && !is.null(unique_predictions) && unique_predictions != "") {
      # If all models' initial predictions are consistent, use this as the final result
      consistent_prediction <- clean_annotation(unique_predictions[1])

      # Check if final annotation differs significantly from consistent initial prediction
      if (char_cluster_id %in% names(final_annotations)) {
        final_prediction <- final_annotations[[char_cluster_id]]

        # If final annotation differs from consistent initial prediction, log warning and correct
        if (!is.null(final_prediction) && final_prediction != consistent_prediction) {
          log_warn(sprintf("Cluster %s has consistent initial predictions but different final annotation, corrected", char_cluster_id), list(
            cluster_id = char_cluster_id,
            consistent_prediction = consistent_prediction,
            final_prediction = final_prediction
          ))
          final_annotations[[char_cluster_id]] <- consistent_prediction
        }
      } else {
        # If no final annotation exists, use consistent initial prediction
        final_annotations[[char_cluster_id]] <- consistent_prediction
      }
    }
  }

  # No need to convert cluster IDs, use original IDs directly
  # This ensures consistency with Seurat's 0-based indexing

  # Return combined results with original cluster IDs
  return(list(
    initial_results = list(
      individual_predictions = initial_results$individual_predictions,
      consensus_results = controversy_results$consensus_results,
      controversial_clusters = controversy_results$controversial_clusters
    ),
    final_annotations = final_annotations,
    controversial_clusters = controversy_results$controversial_clusters,
    discussion_logs = discussion_results$discussion_logs,
    session_id = get_logger()$session_id
  ))
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

# Constants are now defined as function parameters

#' Interactive consensus building for cell type annotation
#'
#' This function implements an interactive voting and discussion mechanism where multiple LLMs
#' collaborate to reach a consensus on cell type annotations, particularly focusing on
#' clusters with low agreement. The process includes:
#' 1. Initial voting by all LLMs
#' 2. Identification of controversial clusters
#' 3. Detailed discussion for controversial clusters
#' 4. Final summary by a designated LLM (default: Claude)
#'
#' @param input Either a data frame from Seurat's FindAllMarkers() function containing 
#'   differential gene expression results (must have columns: 'cluster', 'gene', 
#'   and 'avg_log2FC'), or a list where each element has a 'genes' field containing 
#'   marker genes for a cluster. Cluster IDs must be numeric starting from 0.
#' @param tissue_name Character string specifying the tissue type for context-aware 
#'   cell type annotation. If NULL, generic cell type annotation will be performed.
#' @param models Character vector of model names to use for consensus annotation. 
#'   Minimum 2 models required. Supports models from OpenAI, Anthropic, DeepSeek, 
#'   Google, Alibaba, Stepfun, Zhipu, MiniMax, X.AI, and OpenRouter.
#' @param api_keys Named list of API keys. Can use provider names as keys
#'   (e.g., "openai", "anthropic") or model names as keys (e.g., "gpt-5").
#' @param top_gene_count Integer specifying the number of top marker genes to use 
#'   for annotation per cluster (default: 10).
#' @param controversy_threshold Numeric value between 0 and 1 for consensus proportion 
#'   threshold. Clusters below this threshold are considered controversial (default: 0.7).
#' @param entropy_threshold Numeric value for entropy threshold. Higher entropy 
#'   indicates more disagreement among models (default: 1.0).
#' @param max_discussion_rounds Integer specifying maximum number of discussion rounds 
#'   for controversial clusters (default: 3).
#' @param consensus_check_model Character string specifying which model to use for 
#'   consensus checking. If NULL, uses the first model from the models list.
#' @param log_dir Character string specifying directory for log files (default: "logs").
#' @param cache_dir Character string or NULL. Cache directory for storing results. 
#'   NULL uses system cache, "local" uses current directory, "temp" uses temporary 
#'   directory, or specify custom path.
#' @param use_cache Logical indicating whether to use caching (default: TRUE).
#' @param base_urls Named list or character string specifying custom API base URLs. 
#'   Useful for proxies or alternative endpoints. If NULL, uses official endpoints.
#' @param clusters_to_analyze Character or numeric vector specifying which clusters 
#'   to analyze. If NULL (default), all clusters are analyzed.
#' @param force_rerun Logical indicating whether to force rerun of all specified 
#'   clusters, ignoring cache. Only affects controversial cluster discussions 
#'   (default: FALSE).
#'
#' @return A list containing:
#'   \itemize{
#'     \item \code{voting_results}: Initial voting results from all models
#'     \item \code{controversial_clusters}: Clusters identified as controversial
#'     \item \code{discussion_results}: Detailed discussion results for controversial clusters
#'     \item \code{final_consensus}: Final consensus annotations for all clusters
#'   }
#' @export
interactive_consensus_annotation <- function(input,
                                           tissue_name = NULL,
                                           models = c("claude-opus-4.5",
                                                     "gpt-5.2",
                                                     "gemini-3-pro",
                                                     "deepseek-r1",
                                                     "grok-4.1"),
                                           api_keys,
                                           top_gene_count = 10,
                                           controversy_threshold = 0.7,
                                           entropy_threshold = 1.0,
                                           max_discussion_rounds = 3,
                                           consensus_check_model = NULL,
                                           log_dir = "logs",
                                           cache_dir = NULL,
                                           use_cache = TRUE,
                                           base_urls = NULL,
                                           clusters_to_analyze = NULL,
                                           force_rerun = FALSE) {

  # Check if there are enough models for discussion (at least 2)
  if (length(models) < 2) {
    stop(paste0("At least 2 models are required for LLM discussion and consensus ",
                "building. Please provide more models or use annotate_cell_types() ",
                "function for single-model annotation."))
  }

  # Check if input is a list with named elements (clusters)
  if (is.list(input) && !is.data.frame(input) && !is.null(names(input))) {
    # Check for non-standard cluster IDs that might cause issues
    cluster_names <- names(input)

    # Check if all cluster IDs are numeric
    numeric_names <- suppressWarnings(as.numeric(cluster_names))
    non_numeric_clusters <- cluster_names[is.na(numeric_names)]

    if (length(non_numeric_clusters) > 0) {
      # There are non-numeric cluster IDs
      display_clusters <- non_numeric_clusters[
        seq_len(min(3, length(non_numeric_clusters)))
      ]

      stop(
        "Detected non-numeric cluster IDs: ",
        paste(display_clusters, collapse = ", "),
        if (length(non_numeric_clusters) > 3) " ... (and others)" else "",
        ". \nCluster IDs must be numeric values starting from 0 ",
        "(e.g., '0', '1', '2').\n",
        "Please rename your clusters to use numeric IDs."
      )
    }

    # Try to convert named numeric indices to check if they start from 0
    numeric_names <- suppressWarnings(as.numeric(names(input)))
    if (!all(is.na(numeric_names))) {
      # Has numeric indices, check if they start from 0
      min_index <- min(numeric_names[!is.na(numeric_names)])
      if (min_index > 0) {
        stop(
          "Cluster indices must start from 0 (0-based indexing). ",
          "Found minimum index: ", min_index,
          ". Please convert your indices to start from 0."
        )
      }
    }
  }

  # Initialize cache manager
  cache_manager <- CacheManager$new(cache_dir)
  
  # Get actual cache directory path (important!)
  actual_cache_dir <- cache_manager$get_cache_dir()

  # Log cache settings - use actual path
  if (use_cache && !force_rerun) {
    cache_msg <- sprintf("Cache enabled. Using cache directory: %s", actual_cache_dir)
    log_info(cache_msg, list(cache_dir = actual_cache_dir))
    message(cache_msg)
  } else if (force_rerun) {
    log_info("Force rerun enabled, cache will be ignored for controversial clusters")
    message("Force rerun enabled. Cache will be ignored for controversial clusters.")
  } else {
    log_info("Cache disabled")
    message("Cache disabled.")
  }

  # Filter clusters if clusters_to_analyze is specified
  if (!is.null(clusters_to_analyze)) {
    # Convert to character for consistent comparison
    clusters_to_analyze <- as.character(clusters_to_analyze)
    
    # Get all available clusters
    available_clusters <- if (is.list(input) && !is.data.frame(input)) {
      names(input)
    } else {
      as.character(unique(input$cluster))
    }
    
    # Check which requested clusters exist
    valid_clusters <- clusters_to_analyze[clusters_to_analyze %in% available_clusters]
    invalid_clusters <- clusters_to_analyze[!clusters_to_analyze %in% available_clusters]
    
    # Warn about non-existent clusters
    if (length(invalid_clusters) > 0) {
      warning(sprintf("The following cluster IDs were not found in the input: %s",
                     paste(invalid_clusters, collapse = ", ")))
    }
    
    # Stop if no valid clusters
    if (length(valid_clusters) == 0) {
      stop("None of the specified clusters exist in the input data.")
    }
    
    # Filter input based on type
    if (is.list(input) && !is.data.frame(input)) {
      # For list input, subset by names
      input <- input[valid_clusters]
    } else {
      # For dataframe input, filter rows
      input <- input[input$cluster %in% valid_clusters, ]
    }
    
    # Log the filtering
    log_info(sprintf("Filtered to analyze %d clusters: %s", 
                    length(valid_clusters), 
                    paste(valid_clusters, collapse = ", ")))
    message(sprintf("Analyzing %d specified clusters: %s", 
                   length(valid_clusters), 
                   paste(valid_clusters, collapse = ", ")))
  }

  # Phase 1: Get initial predictions from all models
  initial_results <- get_initial_predictions(
    input = input,
    tissue_name = tissue_name,
    models = models,
    api_keys = api_keys,
    top_gene_count = top_gene_count,
    base_urls = base_urls
  )

  # Phase 2: Identify controversial clusters
  # If consensus_check_model is NULL, use the first available model from the
  # models list
  if (is.null(consensus_check_model) && length(models) > 0) {
    consensus_check_model <- models[1]
    log_msg <- sprintf("No consensus_check_model specified, using %s",
                       consensus_check_model)
    log_info(log_msg, list(consensus_check_model = consensus_check_model))
  }

  controversy_results <- identify_controversial_clusters(
    input = input,
    individual_predictions = initial_results$individual_predictions,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    api_keys = api_keys,
    consensus_check_model = consensus_check_model
  )

  # Phase 3: Process controversial clusters through discussion
  discussion_results <- process_controversial_clusters(
    controversial_clusters = controversy_results$controversial_clusters,
    input = input,
    tissue_name = tissue_name,
    successful_models = initial_results$successful_models,
    api_keys = api_keys,
    individual_predictions = initial_results$individual_predictions,
    top_gene_count = top_gene_count,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    max_discussion_rounds = max_discussion_rounds,
    # No logger parameter needed,
    cache_manager = cache_manager,
    use_cache = use_cache,
    consensus_check_model = consensus_check_model,
    force_rerun = force_rerun
  )

  # Combine results from all phases
  final_results <- combine_results(
    initial_results = initial_results,
    controversy_results = controversy_results,
    discussion_results = discussion_results
  )

  # Print summary of consensus building process
  print_consensus_summary(final_results)

  # Return results
  return(final_results)
}
