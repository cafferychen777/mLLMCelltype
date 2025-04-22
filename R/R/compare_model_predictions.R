#' Compare predictions from different models
#' 
#' This function runs the same input through multiple models and compares their predictions.
#' It provides both individual predictions and a consensus analysis.
#' 
#' @note This function uses create_standardization_prompt from prompt_templates.R
#' @param input Either the differential gene table returned by Seurat FindAllMarkers() function, or a list of genes.
#' @param tissue_name Required. The tissue type or cell source (e.g., 'human PBMC', 'mouse brain', etc.).
#' @param models Vector of model names to compare. Default includes one model from each provider.
#'   Supported models:
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
#' @param top_gene_count Number of top differential genes to be used if input is Seurat differential genes.
#' @param consensus_threshold Minimum proportion of models that must agree for a consensus (default 0.5).
#' @return A list containing individual predictions, consensus results, and agreement statistics.
#' @export
#' @examples
#' \dontrun{
#' # Compare predictions using different models
#' api_keys <- list(
#'   "claude-3-5-sonnet-latest" = "your-anthropic-key",
#'   "deepseek-reasoner" = "your-deepseek-key",
#'   "gemini-1.5-pro" = "your-gemini-key",
#'   "qwen-max-2025-01-25" = "your-qwen-key"
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
                                      models = c("claude-3-7-sonnet-20250219", 
                                                 "claude-3-5-sonnet-latest",
                                                 "claude-3-5-haiku-latest",
                                                 "deepseek-reasoner",
                                                 "deepseek-chat",
                                                 "gemini-2.0-flash",
                                                 "gemini-1.5-pro",
                                                 "qwen-max-2025-01-25",
                                                 "gpt-4o",
                                                 "o1",
                                                 "grok-3-latest"),
                                      api_keys,
                                      top_gene_count = 10,
                                      consensus_threshold = 0.5) {
  
  # Validate inputs
  if (!is.list(api_keys) || length(api_keys) == 0) {
    stop("api_keys must be a non-empty list with named elements corresponding to models")
  }
  
  if (any(!models %in% names(api_keys))) {
    stop("All models must have corresponding API keys in api_keys")
  }
  
  # Initialize results storage
  all_predictions <- list()
  n_clusters <- if(inherits(input, 'list')) length(input) else length(unique(input$cluster))
  successful_models <- character(0)
  
  # Get predictions from each model
  for (model in models) {
    message(sprintf("\nRunning predictions with model: %s", model))
    tryCatch({
      api_key <- get_api_key(model, api_keys)
      
      if (is.null(api_key)) {
        warning(sprintf("No API key found for model '%s' (provider: %s). This model will be skipped.", 
                      model, get_provider(model)))
        next
      }
      
      predictions <- annotate_cell_types(
        input = input,
        tissue_name = tissue_name,
        model = model,
        api_key = api_key,
        top_gene_count = top_gene_count
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
  standardized_predictions <- standardize_cell_type_names(all_predictions, successful_models, api_keys)
  
  # Create comparison matrix only for successful models using standardized predictions
  comparison_matrix <- do.call(cbind, lapply(standardized_predictions[successful_models], function(x) x))
  colnames(comparison_matrix) <- successful_models
  
  # Calculate consensus and agreement statistics
  consensus_results <- apply(comparison_matrix, 1, function(row) {
    # Remove NAs
    valid_predictions <- row[!is.na(row)]
    if (length(valid_predictions) == 0) return(list(consensus = NA, consensus_proportion = NA, entropy = NA))
    
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
        valid_comparisons <- !is.na(all_predictions[[successful_models[i]]]) & 
          !is.na(all_predictions[[successful_models[j]]])
        if (any(valid_comparisons)) {
          agreement <- mean(all_predictions[[successful_models[i]]][valid_comparisons] == 
                              all_predictions[[successful_models[j]]][valid_comparisons])
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
  cat("\nModel Comparison Summary:\n")
  cat(sprintf("Total clusters analyzed: %d\n", summary_stats$total_clusters))
  cat(sprintf("Clusters with consensus: %d (%.1f%%)\n", 
              summary_stats$consensus_reached,
              100 * summary_stats$consensus_reached / summary_stats$total_clusters))
  cat(sprintf("Mean consensus proportion: %.2f\n", summary_stats$mean_consensus_proportion))
  cat(sprintf("Mean entropy: %.2f\n", summary_stats$mean_entropy))
  
  cat("\nPairwise Model Agreement:\n")
  print(model_agreement_matrix)
  
  cat("\nDetailed Results:\n")
  for (i in 1:n_clusters) {
    cat(sprintf("\nCluster %d:\n", i))
    for (model in successful_models) {
      cat(sprintf("  %s: %s (Standardized: %s)\n", 
                model, 
                all_predictions[[model]][i],
                standardized_predictions[[model]][i]))
    }
    cat(sprintf("  Consensus: %s (Consensus Proportion: %.2f, Entropy: %.2f)\n", 
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
#' @param predictions List of predictions from different models
#' @param models Vector of model names that successfully completed predictions
#' @param api_keys Named list of API keys. Can be provided in two formats:
#'   1. With provider names as keys: `list("openai" = "sk-...", "anthropic" = "sk-ant-...", "openrouter" = "sk-or-...")`
#'   2. With model names as keys: `list("gpt-4o" = "sk-...", "claude-3-opus" = "sk-ant-...")`
#' @param standardization_model Model to use for standardization (default: "claude-3-5-sonnet-latest")
#' @return List of standardized predictions with the same structure as the input
#' @keywords internal
standardize_cell_type_names <- function(predictions, 
                                       models, 
                                       api_keys, 
                                       standardization_model = "claude-3-5-sonnet-latest") {
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
      api_key = api_key
    )
    
    # Parse the response to extract mappings
    mapping_lines <- strsplit(response, "\n")[[1]]
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
    cat("\nCell Type Standardization Mapping:\n")
    for (original in names(mapping)) {
      cat(sprintf("  %s -> %s\n", original, mapping[[original]]))
    }
    
    return(standardized_predictions)
    
  }, error = function(e) {
    warning(sprintf("Error in standardization: %s\nReturning original predictions.", e$message))
    return(predictions)
  })
}
