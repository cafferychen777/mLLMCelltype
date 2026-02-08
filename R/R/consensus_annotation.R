# =============================================================================
# CONSTANTS
# =============================================================================

# Internal sentinel values that indicate processing failures, not real cell types.
# Used by select_best_prediction() and clean_annotation() to avoid leaking
# error signals into user-facing results.
.SENTINEL_VALUES <- c(
  "Parsing_Failed",
  "Insufficient_Responses",
  "Prediction_Missing",
  "Annotation_Missing",
  "Unknown"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Parse text-format model predictions into a named list
#'
#' Handles multiple output formats from LLMs:
#' - "cluster_id: cell_type" format
#' - "1. cell_type" numeric index format
#' - Positional fallback (line index maps to cluster index)
#'
#' @param model_preds Character vector of prediction lines from a model
#' @param all_clusters Optional character vector of cluster IDs for positional fallback
#' @return Named list mapping cluster_id -> cell_type
#' @keywords internal
parse_text_predictions <- function(model_preds, all_clusters = NULL) {
  model_structured <- list()

  for (line in model_preds) {
    if (is.na(line) || is.null(line) || trimws(line) == "") next

    # Try "cluster_id: cell_type" format
    parts <- strsplit(line, ":", fixed = TRUE)[[1]]
    if (length(parts) >= 2) {
      cluster_num <- trimws(parts[1])
      cell_type <- trimws(paste(parts[-1], collapse = ":"))
      model_structured[[cluster_num]] <- cell_type
    } else {
      # Try numeric index format: "1. cell_type", "1- cell_type"
      number_match <- regexpr("^\\s*\\d+[\\.-]?\\s+", line)
      if (number_match > 0) {
        number_part <- substr(line, 1, attr(number_match, "match.length"))
        number <- as.numeric(gsub("[^0-9]", "", number_part))
        cell_type <- trimws(substr(line, attr(number_match, "match.length") + 1, nchar(line)))
        cluster_num <- as.character(number)
        model_structured[[cluster_num]] <- cell_type
      }
    }
  }

  # Positional fallback: if specific clusters still have no prediction, try index-based mapping
  # Use position in all_clusters (not arithmetic on cluster ID) since IDs may be non-contiguous
  if (!is.null(all_clusters)) {
    for (cluster_id in all_clusters) {
      if (!is.null(model_structured[[cluster_id]])) next

      index <- match(cluster_id, all_clusters)
      if (is.na(index) || index < 1 || index > length(model_preds)) next

      potential_cell_type <- trimws(model_preds[index])
      if (is.na(potential_cell_type) || potential_cell_type == "") next

      # Strip "cluster_id:" prefix if present
      if (grepl(":", potential_cell_type, fixed = TRUE)) {
        parts <- strsplit(potential_cell_type, ":", fixed = TRUE)[[1]]
        if (length(parts) >= 2) {
          model_structured[[cluster_id]] <- trimws(paste(parts[-1], collapse = ":"))
        }
      } else {
        # Strip numeric index prefix if present
        number_match <- regexpr("^\\s*\\d+[\\.-]?\\s+", potential_cell_type)
        if (number_match > 0) {
          model_structured[[cluster_id]] <- trimws(substr(potential_cell_type,
            attr(number_match, "match.length") + 1, nchar(potential_cell_type)))
        } else {
          model_structured[[cluster_id]] <- potential_cell_type
        }
      }
    }
  }

  model_structured
}

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
    provider <- tryCatch({
      get_provider(model)
    }, error = function(e) {
      log_warn("Failed to resolve provider for model", list(model = model, error = e$message))
      "unknown"
    })

    api_key <- tryCatch({
      get_api_key(model, api_keys)
    }, error = function(e) {
      log_warn("Failed to resolve API key", list(model = model, error = e$message))
      NULL
    })

    if (is.null(api_key)) {
      warning_msg <- sprintf("No API key found for model '%s' (provider: %s). This model will be skipped.",
                            model, provider)
      warning(warning_msg)
      log_warn(warning_msg, list(model = model, provider = provider))
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
identify_controversial_clusters <- function(input, individual_predictions, controversy_threshold, entropy_threshold, api_keys, consensus_check_model = NULL, base_urls = NULL) {
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

  # Get all cluster IDs for positional fallback in text parsing
  all_clusters <- as.character(clusters)

  # Restructure individual_predictions to be indexed by cluster_id
  structured_predictions <- list()

  for (model_name in names(individual_predictions)) {
    model_preds <- individual_predictions[[model_name]]

    if (is.list(model_preds) && !is.null(names(model_preds))) {
      structured_predictions[[model_name]] <- model_preds
    } else if (is.character(model_preds)) {
      structured_predictions[[model_name]] <- parse_text_predictions(model_preds, all_clusters)
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
    initial_consensus <- check_consensus(valid_predictions, api_keys, controversy_threshold, entropy_threshold, consensus_check_model, base_urls)
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
  majority <- consensus_result$majority_prediction

  # Accept the consensus check result if it is a real cell type
  if (!is.null(majority) &&
      !is.na(majority) &&
      is.character(majority) &&
      nzchar(majority) &&
      !majority %in% .SENTINEL_VALUES) {
    return(majority)
  }

  # Fallback: pick the most frequent real prediction from the models
  real_predictions <- valid_predictions[!valid_predictions %in% .SENTINEL_VALUES]
  if (length(real_predictions) == 0) {
    return("Unknown")
  }

  prediction_counts <- table(real_predictions)
  max_count <- max(prediction_counts)
  most_common <- names(prediction_counts[prediction_counts == max_count])

  if (length(most_common) == 1) {
    return(most_common)
  }
  # Tie-break: longest (most specific) annotation wins
  return(most_common[which.max(nchar(most_common))])
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
                                          cache_manager, use_cache, consensus_check_model = NULL, force_rerun = FALSE,
                                          base_urls = NULL) {

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

    # Generate cache key once (reused for both lookup and save)
    cache_key <- if (use_cache) {
      cache_manager$generate_key(input, successful_models, char_cluster_id, tissue_name, top_gene_count)
    } else {
      NULL
    }

    # Check cache
    cached_result <- NULL
    if (use_cache && !force_rerun) {
      cache_debug <- Sys.getenv("LLMCELLTYPE_DEBUG_CACHE") == "TRUE"

      if (cache_debug) {
        message(sprintf("[DEBUG] Cache check for cluster %s: ", char_cluster_id))
      }

      has_cache <- cache_manager$has_cache(cache_key)

      if (cache_debug) {
        message(sprintf("has_cache = %s", has_cache))
      }

      if (has_cache) {
        log_info(sprintf("Loading cached result for cluster %s", char_cluster_id), list(
          cluster_id = char_cluster_id,
          cache_key = cache_key
        ))
        message(sprintf("Loading cached result for cluster %s", char_cluster_id))

        cached_result <- cache_manager$load_from_cache(cache_key)

        if (cache_debug) {
          message(sprintf("[INFO] Successfully loaded cached result for cluster %s", char_cluster_id))
        }
      }
    } else if (force_rerun) {
      log_info(sprintf("Force rerun enabled, skipping cache for cluster %s", char_cluster_id))
    }

    # Use cached results or perform discussion
    if (!is.null(cached_result)) {
      discussion_result <- cached_result$discussion_log
      final_annotation <- cached_result$annotation

      log_info(sprintf("Using cached result for cluster %s", char_cluster_id), list(
        cluster_id = char_cluster_id
      ))
      message(sprintf("Using cached result for cluster %s", char_cluster_id))
    } else {
      # Perform discussion
      discussion_result <- facilitate_cluster_discussion(
        cluster_id = char_cluster_id,
        input = input,
        tissue_name = tissue_name,
        models = successful_models,
        api_keys = api_keys,
        initial_predictions = individual_predictions,
        top_gene_count = top_gene_count,
        max_rounds = max_discussion_rounds,
        controversy_threshold = controversy_threshold,
        entropy_threshold = entropy_threshold,
        consensus_check_model = consensus_check_model,
        base_urls = base_urls
      )

      # Find the last round that has a consensus_result (the last round may
      # lack one if it broke early due to insufficient valid responses)
      final_prediction <- NULL
      for (r in rev(seq_along(discussion_result$rounds))) {
        cr <- discussion_result$rounds[[r]]$consensus_result
        if (!is.null(cr)) {
          final_prediction <- cr$majority_prediction
          break
        }
      }

      # Extract and clean majority_prediction
      final_annotation <- clean_annotation(final_prediction)

      # Save to cache
      if (use_cache) {
        cache_data <- list(
          annotation = final_annotation,
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
    return("Unknown")
  }

  # Remove numbered prefixes like "1. ", "1: ", "1- ", etc.
  annotation <- gsub("^\\d+[\\.:\\-\\s]+\\s*", "", annotation)
  # Remove "CELL TYPE:" prefix
  annotation <- gsub("^CELL\\s*TYPE[\\s:]*", "", annotation)
  # Final trim of whitespace
  annotation <- trimws(annotation)

  # Normalize sentinel values to a user-friendly fallback
  if (annotation %in% .SENTINEL_VALUES || !nzchar(annotation)) {
    return("Unknown")
  }

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
  # Start with non-controversial cluster annotations
  final_annotations <- controversy_results$final_annotations

  # Merge controversial cluster annotations (already cleaned by process_controversial_clusters)
  for (cluster_id in names(discussion_results$final_annotations)) {
    char_cluster_id <- as.character(cluster_id)
    annotation <- discussion_results$final_annotations[[char_cluster_id]]
    if (is.null(annotation) || is.na(annotation) || annotation %in% .SENTINEL_VALUES) {
      annotation <- "Unknown"
    }
    final_annotations[[char_cluster_id]] <- annotation
  }

  result <- list(
    initial_results = list(
      individual_predictions = initial_results$individual_predictions,
      consensus_results = controversy_results$consensus_results,
      controversial_clusters = controversy_results$controversial_clusters
    ),
    final_annotations = final_annotations,
    controversial_clusters = controversy_results$controversial_clusters,
    discussion_logs = discussion_results$discussion_logs,
    session_id = get_logger()$session_id
  )

  # Backward-compatible aliases.
  result$voting_results <- result$initial_results
  result$discussion_results <- result$discussion_logs
  result$final_consensus <- result$final_annotations

  result
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
#'   and 'avg_log2FC'), or a list where each element is either a character vector
#'   of genes or a list containing a `genes` field.
#' @param tissue_name Character string specifying the tissue type for context-aware
#'   cell type annotation (e.g., 'human PBMC', 'mouse brain'). Required.
#' @param models Character vector of model names to use for consensus annotation. 
#'   Minimum 2 models required. Supports models from OpenAI, Anthropic, DeepSeek, 
#'   Google, Alibaba, Stepfun, Zhipu, MiniMax, X.AI, and OpenRouter.
#' @param api_keys Named, non-empty list of API keys. Can use provider names as keys
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
#' @param log_dir Character scalar specifying directory for log files (default: "logs").
#'   This function reinitializes the session logger with this directory at the start
#'   of each call.
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
#'     \item \code{initial_results}: Initial voting results, consensus checks, and controversial cluster IDs
#'     \item \code{final_annotations}: Final annotations keyed by cluster ID
#'     \item \code{controversial_clusters}: Clusters identified as controversial
#'     \item \code{discussion_logs}: Detailed discussion logs for controversial clusters
#'     \item \code{session_id}: Logger session identifier
#'     \item \code{voting_results}: Backward-compatible alias of \code{initial_results}
#'     \item \code{discussion_results}: Backward-compatible alias of \code{discussion_logs}
#'     \item \code{final_consensus}: Backward-compatible alias of \code{final_annotations}
#'   }
#' @importFrom stats setNames
#' @export
interactive_consensus_annotation <- function(input,
                                           tissue_name,
                                           models = c("claude-opus-4-6-20260205",
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
  if (is.null(tissue_name) || !nzchar(trimws(tissue_name))) {
    stop("tissue_name is required. Specify the tissue type (e.g., 'human PBMC', 'mouse brain').")
  }
  if (!is.character(log_dir) || length(log_dir) != 1 || is.na(log_dir) || !nzchar(log_dir)) {
    stop("log_dir must be a non-empty character scalar")
  }
  if (!is.list(api_keys) || is.null(names(api_keys)) || length(api_keys) == 0) {
    stop("api_keys must be a named, non-empty list")
  }

  initialize_logger(log_dir)
  cluster_name_map <- NULL

  # Check if there are enough models for discussion (at least 2)
  if (length(models) < 2) {
    stop(paste0("At least 2 models are required for LLM discussion and consensus ",
                "building. Please provide more models or use annotate_cell_types() ",
                "function for single-model annotation."))
  }

  # Normalize list input to a canonical cluster->genes mapping to keep
  # contract consistent with annotate_cell_types/create_annotation_prompt.
  if (is.list(input) && !is.data.frame(input)) {
    original_names <- names(input)
    normalized_input <- normalize_cluster_gene_list(input)
    if (!is.null(original_names)) {
      cluster_name_map <- setNames(names(normalized_input), original_names)
    }
    input <- lapply(normalized_input, function(genes) list(genes = genes))
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

    # If list input names were normalized, accept either original or normalized IDs.
    if (!is.null(cluster_name_map)) {
      mapped_ids <- ifelse(
        clusters_to_analyze %in% names(cluster_name_map),
        cluster_name_map[clusters_to_analyze],
        clusters_to_analyze
      )
      clusters_to_analyze <- as.character(mapped_ids)
    }
    
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
  # If consensus_check_model is NULL, use the first model that succeeded in
  # Phase 1 (not models[1], which may have failed)
  if (is.null(consensus_check_model)) {
    consensus_check_model <- initial_results$successful_models[1]
    log_info("No consensus_check_model specified, using first successful model",
             list(consensus_check_model = consensus_check_model))
  }

  controversy_results <- identify_controversial_clusters(
    input = input,
    individual_predictions = initial_results$individual_predictions,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    api_keys = api_keys,
    consensus_check_model = consensus_check_model,
    base_urls = base_urls
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
    cache_manager = cache_manager,
    use_cache = use_cache,
    consensus_check_model = consensus_check_model,
    force_rerun = force_rerun,
    base_urls = base_urls
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
