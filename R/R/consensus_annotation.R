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
#' @param input One of the following:
#'   - A data frame from Seurat's FindAllMarkers() function containing differential gene expression results
#'     (must have columns: 'cluster', 'gene', and 'avg_log2FC'). The function will select the top genes
#'     based on avg_log2FC for each cluster.
#'   - A list where each element has a 'genes' field containing marker genes for a cluster.
#'     This can be in one of these formats:
#'     * Named with cluster IDs: list("0" = list(genes = c(...)), "1" = list(genes = c(...)))
#'     * Named with cell type names: list(t_cells = list(genes = c(...)), b_cells = list(genes = c(...)))
#'     * Unnamed list: list(list(genes = c(...)), list(genes = c(...)))
#'   - For both input types, if cluster IDs are numeric and start from 1, they will be automatically
#'     converted to 0-based indexing (e.g., cluster 1 becomes cluster 0) for consistency.
#' @param tissue_name Optional input of tissue name
#' @param models Vector of model names to participate in the discussion. Supported models:
#'   - OpenAI: 'gpt-4o', 'o1' 
#'   - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
#'   - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
#'   - Google: 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
#'   - Alibaba: 'qwen-max-2025-01-25'
#'   - Stepfun: 'step-2-16k', 'step-2-mini', 'step-1-8k'
#'   - Zhipu: 'glm-4-plus', 'glm-3-turbo'
#'   - MiniMax: 'minimax-text-01'
#'   - X.AI: 'grok-3-latest', 'grok-3', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
#'   - OpenRouter: Provides access to models from multiple providers through a single API. Format: 'provider/model-name'
#'     - OpenAI models: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#'     - Anthropic models: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#'     - Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#'     - Google models: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#'     - Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#'     - Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'
#' @param api_keys Named list of API keys. Can be provided in two formats:
#'   1. With provider names as keys: `list("openai" = "sk-...", "anthropic" = "sk-ant-...", "openrouter" = "sk-or-...")`
#'   2. With model names as keys: `list("gpt-4o" = "sk-...", "claude-3-opus" = "sk-ant-...")`
#'   
#'   The system first tries to find the API key using the provider name. If not found, it then tries using the model name.
#'   Example:
#'   ```r
#'   api_keys <- list(
#'     "openai" = Sys.getenv("OPENAI_API_KEY"),
#'     "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
#'     "openrouter" = Sys.getenv("OPENROUTER_API_KEY"),
#'     "claude-3-opus" = "sk-ant-api03-specific-key-for-opus"
#'   )
#'   ```
#' @param top_gene_count Number of top differential genes to use
#' @param controversy_threshold Consensus proportion threshold (default: 0.7). Clusters with consensus proportion below this value will be marked as controversial
#' @param entropy_threshold Entropy threshold for identifying controversial clusters (default: 1.0)
#' @param max_discussion_rounds Maximum number of discussion rounds for controversial clusters (default: 3)
#' @param consensus_check_model Model to use for consensus checking
#' @param log_dir Directory for storing logs
#' @param cache_dir Directory for storing cache
#' @param use_cache Whether to use cached results
#' @return A list containing consensus results, logs, and annotations
#' @name interactive_consensus_annotation
#' @export
NULL

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Get initial predictions from all models
#'
#' This function retrieves initial cell type predictions from all specified models.
#' It is an internal helper function used by the interactive_consensus_annotation function.
#'
#' @param input Either the differential gene table or a list of genes
#' @param tissue_name The tissue type or cell source
#' @param models Vector of model names to use
#' @param api_keys Named list of API keys
#' @param top_gene_count Number of top differential genes to use
#' @param logger Logger object for recording messages
#' @return A list containing individual predictions and successful models
#' @keywords internal
get_initial_predictions <- function(input, tissue_name, models, api_keys, top_gene_count, logger) {
  logger$log_entry("INFO", "Phase 1: Getting initial predictions from all models...")
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
      logger$log_entry("WARNING", warning_msg)
      next
    }

    tryCatch({
      predictions <- annotate_cell_types(
        input = input,
        tissue_name = tissue_name,
        model = model,
        api_key = api_key,
        top_gene_count = top_gene_count
      )
      individual_predictions[[model]] <- predictions
      successful_models <- c(successful_models, model)
    }, error = function(e) {
      warning_msg <- sprintf("Failed to get predictions from %s: %s", model, e$message)
      warning(warning_msg)
      logger$log_entry("WARNING", warning_msg)
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
#' @param input Either the differential gene table or a list of genes
#' @param individual_predictions List of predictions from each model
#' @param controversy_threshold Threshold for marking clusters as controversial
#' @param entropy_threshold Entropy threshold for identifying controversial clusters
#' @param logger Logger object for recording messages
#' @return A list containing controversial clusters and consensus results
#' @keywords internal
identify_controversial_clusters <- function(input, individual_predictions, controversy_threshold, entropy_threshold, api_keys, logger, consensus_check_model = NULL) {
  logger$log_entry("INFO", "Phase 2: Identifying controversial clusters...")
  message("\nPhase 2: Identifying controversial clusters...")

  # Initialize consensus tracking
  consensus_results <- list()
  controversial_clusters <- character(0)
  final_annotations <- list()

  # For each cluster, check consensus
  clusters <- if (inherits(input, 'list')) {
    names(input)
  } else {
    unique(input$cluster)
  }
  
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
        }
      }
      
      structured_predictions[[model_name]] <- model_structured
    }
  }
  
  for (cluster_id in clusters) {
    # Use original cluster ID for log output, no conversion needed
    logger$log_entry("INFO", sprintf("Analyzing cluster %s...", cluster_id))
    message(sprintf("\nAnalyzing cluster %s...", cluster_id))

    # Get predictions for this cluster from each model
    cluster_predictions <- sapply(structured_predictions, function(x) {
      if (is.null(x[[as.character(cluster_id)]])) NA else x[[as.character(cluster_id)]]
    })
    valid_predictions <- cluster_predictions[!is.na(cluster_predictions)]

    if (length(valid_predictions) == 0) {
      logger$log_entry("WARNING", sprintf("No valid predictions for cluster %s. Marking as controversial.", cluster_id))
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

      logger$log_entry("INFO", sprintf("Cluster %s marked as controversial (reached: %s, consensus proportion: %.2f, entropy: %.2f)",
                        cluster_id, initial_consensus$reached,
                        initial_consensus$consensus_proportion, initial_consensus$entropy))

      message(sprintf("Cluster %s marked as controversial (reached: %s, consensus proportion: %.2f, entropy: %.2f)",
                     cluster_id, initial_consensus$reached,
                     initial_consensus$consensus_proportion, initial_consensus$entropy))

      controversial_clusters <- c(controversial_clusters, as.character(cluster_id))
    } else {
      # Process non-controversial clusters
      final_annotations[[as.character(cluster_id)]] <- select_best_prediction(initial_consensus, valid_predictions)

      logger$log_entry("INFO", sprintf("Consensus reached for cluster %s (consensus proportion: %.2f, entropy: %.2f, selected: %s)",
                        cluster_id, initial_consensus$consensus_proportion,
                        initial_consensus$entropy, final_annotations[[as.character(cluster_id)]]))

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
#' @param consensus_result Consensus analysis result
#' @param valid_predictions Valid predictions for the cluster
#' @return The best prediction
#' @keywords internal
select_best_prediction <- function(consensus_result, valid_predictions) {
  # If we have a majority prediction from Claude, use it
  if (!is.na(consensus_result$majority_prediction) &&
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
#' @param controversial_clusters List of controversial cluster IDs
#' @param input Either the differential gene table or a list of genes
#' @param tissue_name The tissue type or cell source
#' @param successful_models Vector of successful model names
#' @param api_keys Named list of API keys
#' @param individual_predictions List of predictions from each model
#' @param top_gene_count Number of top differential genes to use
#' @param controversy_threshold Threshold for marking clusters as controversial
#' @param max_discussion_rounds Maximum number of discussion rounds for controversial clusters
#' @param logger Logger object for recording messages
#' @param cache_manager Cache manager object
#' @param use_cache Whether to use cached results
#' @return A list containing discussion logs and final annotations
#' @keywords internal
process_controversial_clusters <- function(controversial_clusters, input, tissue_name,
                                          successful_models, api_keys, individual_predictions,
                                          top_gene_count, controversy_threshold, entropy_threshold, max_discussion_rounds,
                                          logger, cache_manager, use_cache) {

  if (length(controversial_clusters) == 0) {
    logger$log_entry("INFO", "No controversial clusters found. All clusters have reached consensus.")
    message("\nNo controversial clusters found. All clusters have reached consensus.")
    return(list(
      discussion_logs = list(),
      final_annotations = list()
    ))
  }

  logger$log_entry("INFO", sprintf("Phase 3: Starting discussions for %d controversial clusters...",
                    length(controversial_clusters)))
  message(sprintf("\nPhase 3: Starting discussions for %d controversial clusters...",
                 length(controversial_clusters)))

  discussion_logs <- list()
  final_annotations <- list()

  for (cluster_id in controversial_clusters) {
    cluster_id_num <- as.numeric(cluster_id)
    logger$log_entry("INFO", sprintf("Starting discussion for cluster %d...", cluster_id_num))
    message(sprintf("\nStarting discussion for cluster %d...", cluster_id_num))

    # Check cache
    cached_result <- NULL
    if (use_cache) {
      cache_key <- cache_manager$generate_key(input, successful_models, cluster_id_num)
      cache_debug <- Sys.getenv("LLMCELLTYPE_DEBUG_CACHE") == "TRUE"

      if (cache_debug) {
        cat(sprintf("[DEBUG] Cache check for cluster %s: ", cluster_id))
      }

      has_cache <- cache_manager$has_cache(cache_key)

      if (cache_debug) {
        cat(sprintf("has_cache = %s\n", has_cache))
      }

      if (has_cache) {
        # Use cached results
        logger$log_entry("INFO", sprintf("Loading cached result for cluster %d", cluster_id_num))
        message(sprintf("Loading cached result for cluster %d", cluster_id_num))

        cached_result <- cache_manager$load_from_cache(cache_key)

        if (cache_debug) {
          cat(sprintf("[INFO] Successfully loaded cached result for cluster %s\n", cluster_id))
        }
      }
    }

    # Use cached results or perform discussion
    if (!is.null(cached_result)) {
      # Use cached results
      discussion_result <- cached_result$discussion_log
      final_annotation <- cached_result$annotation

      logger$log_entry("INFO", sprintf("Using cached result for cluster %d", cluster_id_num))
      message(sprintf("Using cached result for cluster %d", cluster_id_num))
    } else {
      # Perform discussion
      # Parameters are passed through to check_consensus and used in prompt template to instruct LLM # nolint
      discussion_result <- facilitate_cluster_discussion(
        cluster_id = cluster_id_num,
        input = input,
        tissue_name = tissue_name,
        models = successful_models,  # Only use models that worked in initial phase
        api_keys = api_keys,
        initial_predictions = individual_predictions,
        top_gene_count = top_gene_count,
        max_rounds = max_discussion_rounds,
        controversy_threshold = controversy_threshold,
        entropy_threshold = entropy_threshold,
        logger = logger
      )

      # Get results from the last round of discussion
      last_round_index <- length(discussion_result$rounds)
      last_round <- discussion_result$rounds[[last_round_index]]

      # Extract and clean majority_prediction
      final_annotation <- clean_annotation(last_round$consensus_result$majority_prediction)

      # Save to cache - fix cache content structure
      if (use_cache) {
        cache_key <- cache_manager$generate_key(input, successful_models, cluster_id_num)
        cache_data <- list(
          annotation = final_annotation,  # Use the correct final_annotation variable
          discussion_log = discussion_result,
          is_controversial = TRUE
        )
        cache_manager$save_to_cache(cache_key, cache_data)
        logger$log_entry("INFO", sprintf("Saved result to cache for cluster %d", cluster_id_num))
      }
    }

    discussion_logs[[cluster_id]] <- discussion_result
    final_annotations[[cluster_id]] <- final_annotation

    logger$log_entry("INFO", sprintf("Completed discussion for cluster %d", cluster_id_num))
    message(sprintf("Completed discussion for cluster %d", cluster_id_num))
  }

  return(list(
    discussion_logs = discussion_logs,
    final_annotations = final_annotations
  ))
}

#' Clean annotation text by removing prefixes and extra whitespace
#'
#' @param annotation The annotation text to clean
#' @return Cleaned annotation text
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
#' @param initial_results Results from initial prediction phase
#' @param controversy_results Results from controversy identification phase
#' @param discussion_results Results from discussion phase
#' @param logger Logger object
#' @return Combined results
#' @keywords internal
combine_results <- function(initial_results, controversy_results, discussion_results, logger) {
  # Combine final annotations from non-controversial and controversial clusters
  final_annotations <- controversy_results$final_annotations

  # Create a mapping table for discussion results
  discussion_results_map <- list()
  for (cluster_id in names(discussion_results$discussion_logs)) {
    # Get discussion log
    log <- discussion_results$discussion_logs[[cluster_id]]

    # Check discussion rounds
    if (length(log$rounds) > 0) {
      last_round <- log$rounds[[length(log$rounds)]]

      # Check if there is a consensus result
      if ("consensus_result" %in% names(last_round)) {
        majority_prediction <- last_round$consensus_result$majority_prediction
        discussion_results_map[[cluster_id]] <- majority_prediction
      }
    }
  }

  # Update final annotations using the mapping table
  for (cluster_id in names(discussion_results_map)) {
    final_annotations[[cluster_id]] <- discussion_results_map[[cluster_id]]
  }

  # Verify consistency between final annotations and discussion results
  for (cluster_id in names(discussion_results_map)) {
    if (cluster_id %in% names(final_annotations)) {
      # First handle NA values with descriptive replacements
      if (is.na(final_annotations[[cluster_id]])) {
        logger$log_entry("WARNING", sprintf("Cluster %s final annotation is NA, replacing with descriptive placeholder", cluster_id))
        final_annotations[[cluster_id]] <- "Data_Missing_In_Final_Annotations"
      }
      if (is.na(discussion_results_map[[cluster_id]])) {
        logger$log_entry("WARNING", sprintf("Cluster %s discussion result is NA, replacing with descriptive placeholder", cluster_id))
        discussion_results_map[[cluster_id]] <- "Data_Missing_In_Discussion_Results"
      }
      
      # Now we can safely compare without NA concerns
      if (final_annotations[[cluster_id]] != discussion_results_map[[cluster_id]]) {
        logger$log_entry("WARNING", sprintf("Cluster %s final annotation '%s' differs from discussion result '%s', corrected",
                                          cluster_id,
                                          final_annotations[[cluster_id]],
                                          discussion_results_map[[cluster_id]]))
        final_annotations[[cluster_id]] <- discussion_results_map[[cluster_id]]
      }
    }
  }

  # Check consistency with initial predictions
  for (cluster_id in controversy_results$controversial_clusters) {
    # Collect all initial predictions
    initial_predictions <- list()
    for (model in names(initial_results$individual_predictions)) {
      if (cluster_id %in% names(initial_results$individual_predictions[[model]])) {
        prediction <- initial_results$individual_predictions[[model]][[cluster_id]]
        initial_predictions[[model]] <- prediction
      }
    }

    # Check if initial predictions are consistent
    unique_predictions <- unique(unlist(initial_predictions))
    if (length(unique_predictions) == 1 && !is.null(unique_predictions) && unique_predictions != "") {
      # If all models' initial predictions are consistent, use this as the final result
      consistent_prediction <- clean_annotation(unique_predictions[1])

      # Check if final annotation differs significantly from consistent initial prediction
      if (cluster_id %in% names(final_annotations)) {
        final_prediction <- final_annotations[[cluster_id]]

        # If final annotation differs from consistent initial prediction, log warning and correct
        if (!is.null(final_prediction) && final_prediction != consistent_prediction) {
          logger$log_entry("WARNING", sprintf("Cluster %s has consistent initial predictions '%s' but final annotation is '%s', corrected",
                                            cluster_id,
                                            consistent_prediction,
                                            final_prediction))
          final_annotations[[cluster_id]] <- consistent_prediction
        }
      } else {
        # If no final annotation exists, use consistent initial prediction
        final_annotations[[cluster_id]] <- consistent_prediction
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
    session_id = logger$session_id
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
#' @param input One of the following:
#'   - A data frame from Seurat's FindAllMarkers() function containing differential gene expression results
#'     (must have columns: 'cluster', 'gene', and 'avg_log2FC'). The function will select the top genes
#'     based on avg_log2FC for each cluster.
#'   - A list where each element has a 'genes' field containing marker genes for a cluster.
#'     This can be in one of these formats:
#'     * Named with cluster IDs: list("0" = list(genes = c(...)), "1" = list(genes = c(...)))
#'     * Named with cell type names: list(t_cells = list(genes = c(...)), b_cells = list(genes = c(...)))
#'     * Unnamed list: list(list(genes = c(...)), list(genes = c(...)))
#'   - For both input types, if cluster IDs are numeric and start from 1, they will be automatically
#'     converted to 0-based indexing (e.g., cluster 1 becomes cluster 0) for consistency.
#' @param tissue_name Optional input of tissue name
#' @param models Vector of model names to participate in the discussion. Supported models:
#'   - OpenAI: 'gpt-4o', 'o1' 
#'   - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
#'   - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
#'   - Google: 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
#'   - Alibaba: 'qwen-max-2025-01-25'
#'   - Stepfun: 'step-2-16k', 'step-2-mini', 'step-1-8k'
#'   - Zhipu: 'glm-4-plus', 'glm-3-turbo'
#'   - MiniMax: 'minimax-text-01'
#'   - X.AI: 'grok-3-latest', 'grok-3', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
#'   - OpenRouter: Provides access to models from multiple providers through a single API. Format: 'provider/model-name'
#'     - OpenAI models: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#'     - Anthropic models: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#'     - Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#'     - Google models: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#'     - Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#'     - Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'
#' @param api_keys Named list of API keys. Can be provided in two formats:
#'   1. With provider names as keys: `list("openai" = "sk-...", "anthropic" = "sk-ant-...", "openrouter" = "sk-or-...")`
#'   2. With model names as keys: `list("gpt-4o" = "sk-...", "claude-3-opus" = "sk-ant-...")`
#'   
#'   The system first tries to find the API key using the provider name. If not found, it then tries using the model name.
#'   Example:
#'   ```r
#'   api_keys <- list(
#'     "openai" = Sys.getenv("OPENAI_API_KEY"),
#'     "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
#'     "openrouter" = Sys.getenv("OPENROUTER_API_KEY"),
#'     "claude-3-opus" = "sk-ant-api03-specific-key-for-opus"
#'   )
#'   ```
#' @param top_gene_count Number of top differential genes to use
#' @param controversy_threshold Consensus proportion threshold (default: 0.7). Clusters with consensus proportion below this value will be marked as controversial
#' @param entropy_threshold Entropy threshold for identifying controversial clusters (default: 1.0)
#' @param max_discussion_rounds Maximum number of discussion rounds for controversial clusters (default: 3)
#' @param consensus_check_model Model to use for consensus checking
#' @param log_dir Directory for storing logs
#' @param cache_dir Directory for storing cache
#' @param use_cache Whether to use cached results
#' @return A list containing consensus results, logs, and annotations
#' @export
interactive_consensus_annotation <- function(input,
                                           tissue_name = NULL,
                                           models = c("claude-3-7-sonnet-20250219",
                                                     "claude-3-5-sonnet-latest",
                                                     "claude-3-5-haiku-latest",
                                                     "gemini-2.0-flash",
                                                     "gemini-1.5-pro",
                                                     "qwen-max-2025-01-25",
                                                     "gpt-4o",
                                                     "grok-3-latest"),
                                           api_keys,
                                           top_gene_count = 10,
                                           controversy_threshold = 0.7,
                                           entropy_threshold = 1.0,
                                           max_discussion_rounds = 3,
                                           consensus_check_model = NULL,
                                           log_dir = "logs",
                                           cache_dir = "consensus_cache",
                                           use_cache = TRUE) {

  # Check if there are enough models for discussion (at least 2)
  if (length(models) < 2) {
    stop("At least 2 models are required for LLM discussion and consensus building. Please provide more models.")
  }

  # Initialize logger and cache manager
  logger <- DiscussionLogger$new(log_dir)
  cache_manager <- CacheManager$new(cache_dir)

  # Log cache settings
  if (use_cache) {
    logger$log_entry("INFO", sprintf("Cache enabled. Using cache directory: %s", cache_dir))
    message(sprintf("Cache enabled. Using cache directory: %s", cache_dir))
  } else {
    logger$log_entry("INFO", "Cache disabled.")
    message("Cache disabled.")
  }

  # Phase 1: Get initial predictions from all models
  initial_results <- get_initial_predictions(
    input = input,
    tissue_name = tissue_name,
    models = models,
    api_keys = api_keys,
    top_gene_count = top_gene_count,
    logger = logger
  )

  # Phase 2: Identify controversial clusters
  # If consensus_check_model is NULL, use the first available model from the models list
  if (is.null(consensus_check_model) && length(models) > 0) {
    consensus_check_model <- models[1]
    logger$log_entry("INFO", sprintf("No consensus_check_model specified, using %s", consensus_check_model))
  }
  
  controversy_results <- identify_controversial_clusters(
    input = input,
    individual_predictions = initial_results$individual_predictions,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    api_keys = api_keys,
    logger = logger,
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
    logger = logger,
    cache_manager = cache_manager,
    use_cache = use_cache
  )

  # Combine results from all phases
  final_results <- combine_results(
    initial_results = initial_results,
    controversy_results = controversy_results,
    discussion_results = discussion_results,
    logger = logger
  )

  # Print summary of consensus building process
  print_consensus_summary(final_results)

  # Return results
  return(final_results)
}
