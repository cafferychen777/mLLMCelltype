#' Compare predictions from different models
#' 
#' This function runs the same input through multiple models and compares their predictions.
#' It provides both individual predictions and a consensus analysis.
#' 
#' @note This function uses create_standardization_prompt from prompt_templates.R
#
#
#
#'   Supported models:
#'   - OpenAI: 'gpt-5.2', 'gpt-5.1', 'gpt-5', 'gpt-4.1', 'gpt-4o', 'o3-pro', 'o3', 'o4-mini', 'o1', 'o1-pro'
#'   - Anthropic: 'claude-opus-4-6-20260205', 'claude-opus-4-5-20251101', 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001',
#'     'claude-opus-4-1-20250805', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219'
#'   - DeepSeek: 'deepseek-chat', 'deepseek-reasoner', 'deepseek-r1'
#'   - Google: 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash'
#'   - Alibaba: 'qwen3-max', 'qwen-max-2025-01-25', 'qwen-plus'
#'   - Stepfun: 'step-3', 'step-2-16k', 'step-2-mini'
#'   - Zhipu: 'glm-4.7', 'glm-4-plus'
#'   - MiniMax: 'minimax-m2.1', 'minimax-m2', 'MiniMax-Text-01'
#'   - X.AI: 'grok-4', 'grok-4.1', 'grok-4-heavy', 'grok-3', 'grok-3-fast', 'grok-3-mini'
#'   - OpenRouter: Provides access to models from multiple providers through a single API. Format: 'provider/model-name'
#'     - OpenAI models: 'openai/gpt-5.2', 'openai/gpt-5', 'openai/o3-pro', 'openai/o4-mini'
#'     - Anthropic models: 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5'
#'     - Meta models: 'meta-llama/llama-4-maverick', 'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct'
#'     - Google models: 'google/gemini-3-pro', 'google/gemini-3-flash', 'google/gemini-2.5-pro'
#'     - Mistral models: 'mistralai/mistral-large', 'mistralai/magistral-medium-2506'
#'     - Other models: 'deepseek/deepseek-r1', 'deepseek/deepseek-chat-v3.1', 'microsoft/mai-ds-r1'
#
#'   1. With provider names as keys: `list("openai" = "sk-...", "anthropic" = "sk-ant-...", "openrouter" = "sk-or-...")`
#'   2. With model names as keys: `list("gpt-5" = "sk-...", "claude-sonnet-4-5-20250929" = "sk-ant-...")`
#'   
#'   The system first tries to find the API key using the provider name. If not found, it then tries using the model name.
#'   Example:
#'   ```r
#'   api_keys <- list(
#'     "openai" = Sys.getenv("OPENAI_API_KEY"),
#'     "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
#'     "openrouter" = Sys.getenv("OPENROUTER_API_KEY"),
#'     "claude-opus-4-6-20260205" = "sk-ant-api03-specific-key-for-opus"
#'   )
#'   ```
#'
#' @param input Either a data frame from Seurat's FindAllMarkers() containing columns 'cluster', 'gene', and 'avg_log2FC', or a list with 'genes' field for each cluster
#' @param tissue_name Tissue context (e.g., 'human PBMC', 'mouse brain') for more accurate annotations
#' @param models Vector of model names to use for comparison. Default includes top models from each provider
#' @param api_keys Named list of API keys for the models, with provider or model names as keys.
#'   Every model in \code{models} must resolve to a non-NULL API key.
#' @param top_gene_count Number of top genes to use per cluster when input is from Seurat. Default: 10
#' @param consensus_threshold Minimum agreement threshold for consensus (0-1). Default: 0.5.
#'   Consensus is only evaluated when at least two non-missing model predictions are available for a cluster.
#' @param base_urls Optional base URLs for API endpoints. Can be a string or named list for provider-specific custom endpoints.
#'
#' @return List containing individual model predictions and consensus analysis
#'   If a cluster has fewer than two valid predictions after alignment/padding,
#'   its consensus-related outputs are \code{NA}.
#' @export
#' @examples
#' \dontrun{
#' # Compare predictions using different models
#' api_keys <- list(
#'   "claude-sonnet-4-5-20250929" = "your-anthropic-key",
#'   "deepseek-reasoner" = "your-deepseek-key",
#'   "gemini-3-pro" = "your-gemini-key",
#'   "qwen3-max" = "your-qwen-key"
#' )
#' 
#' results <- compare_model_predictions(
#'   input = list(gs1=c('CD4','CD3D'), gs2='CD14'),
#'   tissue_name = 'PBMC',
#'   api_keys = api_keys
#' )
#' }
compare_model_predictions <- function(input,
                                      tissue_name,
                                      models = c("claude-opus-4-6-20260205",
                                                 "gpt-5.2",
                                                 "gemini-3-pro",
                                                 "deepseek-r1",
                                                 "o3-pro",
                                                 "grok-4.1"),
                                      api_keys,
                                      top_gene_count = 10,
                                      consensus_threshold = 0.5,
                                      base_urls = NULL) {
  
  # Validate inputs
  if (!is.list(api_keys) || length(api_keys) == 0) {
    stop("api_keys must be a non-empty list with named elements corresponding to models")
  }
  
  # Validate that each model can resolve an API key (via provider name or model name)
  for (m in models) {
    if (is.null(get_api_key(m, api_keys))) {
      stop(sprintf("No API key found for model '%s'. Provide a key using the provider name '%s' or the model name as the key.",
                   m, get_provider(m)))
    }
  }
  
  # Extract cluster IDs from input for display
  prompt_result <- create_annotation_prompt(input, tissue_name, top_gene_count)
  cluster_ids <- names(prompt_result$gene_lists)

  # Initialize results storage
  all_predictions <- list()
  successful_models <- character(0)

  # Get predictions from each model
  for (model in models) {
    message(sprintf("\nRunning predictions with model: %s", model))
    tryCatch({
      api_key <- get_api_key(model, api_keys)

      predictions <- annotate_cell_types(
        input = input,
        tissue_name = tissue_name,
        model = model,
        api_key = api_key,
        top_gene_count = top_gene_count,
        base_urls = base_urls
      )
      all_predictions[[model]] <- predictions
      successful_models <- c(successful_models, model)
    }, error = function(e) {
      warning(sprintf("Error with model %s: %s", model, e$message))
    })
  }
  
  # Check if we have any successful predictions
  if (length(successful_models) == 0) {
    stop("No models successfully completed predictions")
  }
  
  # Standardize cell type names using LLM
  message("\nStandardizing cell type names...")
  standardized_predictions <- standardize_cell_type_names(all_predictions, successful_models, api_keys, base_urls = base_urls)
  
  # Pad all prediction vectors to equal length with NA to avoid vector recycling
  all_vectors <- all_predictions[successful_models]
  std_vectors <- standardized_predictions[successful_models]
  max_len <- max(vapply(std_vectors, length, integer(1)),
                 vapply(all_vectors, length, integer(1)))
  pad <- function(x) { length(x) <- max_len; x }

  comparison_matrix <- do.call(cbind, lapply(std_vectors, pad))
  colnames(comparison_matrix) <- successful_models
  raw_matrix <- do.call(cbind, lapply(all_vectors, pad))
  colnames(raw_matrix) <- successful_models
  n_clusters <- nrow(comparison_matrix)
  
  # Calculate consensus and agreement statistics
  consensus_results <- apply(comparison_matrix, 1, function(row) {
    # Remove NAs
    valid_predictions <- row[!is.na(row)]
    if (length(valid_predictions) < 2) {
      return(list(consensus = NA, consensus_proportion = NA, entropy = NA))
    }
    
    # Count occurrences of each prediction
    pred_table <- table(valid_predictions)
    max_agreement <- max(pred_table) / length(valid_predictions)
    
    # Get consensus if agreement meets threshold
    consensus <- if (max_agreement >= consensus_threshold) {
      names(pred_table)[which.max(pred_table)]
    } else {
      NA
    }
    
    # Calculate consensus proportion
    consensus_proportion <- if (!is.na(consensus)) {
      pred_table[consensus] / length(valid_predictions)
    } else {
      NA
    }
    
    # Calculate entropy
    entropy <- -sum(pred_table / length(valid_predictions) * log2(pred_table / length(valid_predictions)))
    
    list(
      consensus = consensus,
      consensus_proportion = consensus_proportion,
      entropy = entropy
    )
  })
  
  # Format results
  consensus_predictions <- sapply(consensus_results, function(x) x$consensus)
  consensus_proportions <- sapply(consensus_results, function(x) x$consensus_proportion)
  entropies <- sapply(consensus_results, function(x) x$entropy)
  
  # Calculate overall statistics
  model_agreement_matrix <- matrix(NA, 
                                   nrow = length(successful_models), 
                                   ncol = length(successful_models),
                                   dimnames = list(successful_models, successful_models))
  
  for (i in seq_along(successful_models)) {
    for (j in seq_along(successful_models)) {
      if (i != j) {
        valid_comparisons <- !is.na(comparison_matrix[, successful_models[i]]) &
          !is.na(comparison_matrix[, successful_models[j]])
        if (any(valid_comparisons)) {
          agreement <- mean(comparison_matrix[valid_comparisons, successful_models[i]] ==
                              comparison_matrix[valid_comparisons, successful_models[j]])
          model_agreement_matrix[i,j] <- agreement
        }
      }
    }
  }
  
  # Prepare summary statistics
  summary_stats <- list(
    total_clusters = n_clusters,
    consensus_reached = sum(!is.na(consensus_predictions)),
    mean_consensus_proportion = mean(consensus_proportions, na.rm = TRUE),
    mean_entropy = mean(entropies, na.rm = TRUE),
    model_agreement_matrix = model_agreement_matrix
  )
  
  # Return results
  results <- list(
    individual_predictions = all_predictions[successful_models],
    standardized_predictions = standardized_predictions,
    comparison_matrix = comparison_matrix,
    consensus_predictions = consensus_predictions,
    consensus_proportions = consensus_proportions,
    entropies = entropies,
    summary_stats = summary_stats
  )
  
  # Print summary
  message("\nModel Comparison Summary:\n")
  message(sprintf("Total clusters analyzed: %d\n", summary_stats$total_clusters))
  message(sprintf("Clusters with consensus: %d (%.1f%%)\n", 
              summary_stats$consensus_reached,
              100 * summary_stats$consensus_reached / summary_stats$total_clusters))
  message(sprintf("Mean consensus proportion: %.2f\n", summary_stats$mean_consensus_proportion))
  message(sprintf("Mean entropy: %.2f\n", summary_stats$mean_entropy))
  
  message("\nPairwise Model Agreement:\n")
  print(model_agreement_matrix)
  
  message("\nDetailed Results:\n")
  for (i in 1:n_clusters) {
    cluster_label <- if (i <= length(cluster_ids)) cluster_ids[i] else as.character(i)
    message(sprintf("\nCluster %s:\n", cluster_label))
    for (model in successful_models) {
      message(sprintf("  %s: %s (Standardized: %s)\n",
                model,
                raw_matrix[i, model],
                comparison_matrix[i, model]))
    }
    message(sprintf("  Consensus: %s (Consensus Proportion: %.2f, Entropy: %.2f)\n", 
                consensus_predictions[i], 
                consensus_proportions[i],
                entropies[i]))
  }
  
  invisible(results)
}

#' Standardize cell type names using a language model
#' 
#' This function takes predictions from multiple models and standardizes the cell type
#' nomenclature to ensure consistent naming across different models' outputs.
#' 
#
#
#
#'   1. With provider names as keys: `list("openai" = "sk-...", "anthropic" = "sk-ant-...", "openrouter" = "sk-or-...")`
#'   2. With model names as keys: `list("gpt-5" = "sk-...", "claude-sonnet-4-5-20250929" = "sk-ant-...")`
#
#
#' @keywords internal
standardize_cell_type_names <- function(predictions,
                                       models,
                                       api_keys,
                                       standardization_model = "claude-sonnet-4-20250514",
                                       base_urls = NULL) {
  # Get API key for standardization model
  api_key <- get_api_key(standardization_model, api_keys)
  
  if (is.null(api_key)) {
    warning(sprintf("No API key found for standardization model '%s'. Using the first available model instead.", 
                  standardization_model))
    standardization_model <- models[1]
    api_key <- get_api_key(standardization_model, api_keys)
  }
  
  # Get unique cell type names from all predictions
  all_cell_types <- unique(unlist(predictions[models]))
  all_cell_types <- all_cell_types[!is.na(all_cell_types)]
  
  if (length(all_cell_types) == 0) {
    warning("No valid cell type predictions found to standardize")
    return(predictions)
  }
  
  # Create a mapping of original cell types to standardized names
  message(sprintf("Using %s to standardize %d unique cell type names", 
                standardization_model, length(all_cell_types)))
  
  # Use the standardization prompt template from prompt_templates.R
  prompt <- create_standardization_prompt(all_cell_types)
  
  # Call the LLM to get standardized names
  tryCatch({
    response <- get_model_response(
      prompt = prompt,
      model = standardization_model,
      api_key = api_key,
      base_urls = base_urls
    )
    
    # Parse the response to extract mappings
    # response may be a character vector (one element per line from get_model_response),
    # so collapse first to ensure all lines are processed
    mapping_lines <- strsplit(paste(response, collapse = "\n"), "\n")[[1]]
    mapping <- list()
    
    for (line in mapping_lines) {
      # Skip empty lines
      if (nchar(trimws(line)) == 0) next
      
      # Extract original and standardized names
      parts <- strsplit(line, ":")[[1]]
      if (length(parts) >= 2) {
        original <- trimws(parts[1])
        standardized <- trimws(paste(parts[-1], collapse = ":"))
        mapping[[original]] <- standardized
      }
    }
    
    # Function to clean cell type names by removing prefixes, numbers, etc.
    clean_cell_type <- function(cell_type) {
      if (is.na(cell_type)) return(cell_type)
      
      # Remove various number/cluster prefixes
      cleaned <- cell_type
      # Remove "1: ", "Cluster 1: ", etc.
      cleaned <- gsub("^\\s*\\d+\\s*:\\s*", "", cleaned)
      cleaned <- gsub("^\\s*[Cc]luster\\s*\\d+\\s*:\\s*", "", cleaned)
      # Also remove any leading numbers with any separator
      cleaned <- gsub("^\\s*\\d+\\s*[.:-]?\\s*", "", cleaned)
      # Trim any leading/trailing whitespace
      cleaned <- trimws(cleaned)
      
      return(cleaned)
    }
    
    # Apply standardization to all predictions
    standardized_predictions <- predictions
    for (model in models) {
      for (i in seq_along(predictions[[model]])) {
        original <- predictions[[model]][i]
        if (!is.na(original)) {
          # Clean the original name first
          cleaned_original <- clean_cell_type(original)
          
          # Try direct mapping with original or cleaned name
          if (original %in% names(mapping)) {
            standardized <- mapping[[original]]
          } else if (cleaned_original %in% names(mapping)) {
            standardized <- mapping[[cleaned_original]]
          } else {
            # If no mapping found, use the cleaned original
            standardized <- cleaned_original
          }
          
          # Clean the standardized result as well to remove any remaining prefixes
          standardized_predictions[[model]][i] <- clean_cell_type(standardized)
        }
      }
    }
    
    # Print standardization mapping for reference
    message("\nCell Type Standardization Mapping:\n")
    for (original in names(mapping)) {
      message(sprintf("  %s -> %s\n", original, mapping[[original]]))
    }
    
    return(standardized_predictions)
    
  }, error = function(e) {
    warning(sprintf("Error in standardization: %s\nReturning original predictions.", e$message))
    return(predictions)
  })
}
