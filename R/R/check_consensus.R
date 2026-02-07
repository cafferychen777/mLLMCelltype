
#' Normalize annotation for comparison
#
#
#' @keywords internal
normalize_annotation <- function(annotation) {
  # Convert to lowercase
  normalized <- tolower(trimws(annotation))
  
  # Replace punctuation with space to preserve word boundaries (e.g., "T-cell" -> "T cell")
  normalized <- gsub("[[:punct:]]", " ", normalized)
  # Normalize whitespace
  normalized <- gsub("\\s+", " ", normalized)
  normalized <- trimws(normalized)
  
  # Handle plurals (simple approach)
  normalized <- gsub("cells$", "cell", normalized)
  normalized <- gsub("cytes$", "cyte", normalized)
  
  # Common cell type variations
  normalized <- gsub("t lymphocyte", "t cell", normalized)
  normalized <- gsub("b lymphocyte", "b cell", normalized)
  normalized <- gsub("natural killer", "nk", normalized)
  
  return(normalized)
}

#' Calculate simple consensus without LLM
#
#
#' @keywords internal
calculate_simple_consensus <- function(round_responses) {
  # Normalize all annotations
  normalized_responses <- sapply(round_responses, normalize_annotation)
  
  # Count occurrences
  response_counts <- table(normalized_responses)
  total_responses <- length(round_responses)
  
  
  # Find most common response
  most_common_idx <- which.max(response_counts)
  most_common_normalized <- names(response_counts)[most_common_idx]
  most_common_count <- as.numeric(response_counts[most_common_idx])
  
  # Find original annotation that matches the most common normalized form
  matching_idx <- which(normalized_responses == most_common_normalized)[1]
  majority_prediction <- round_responses[matching_idx]
  
  # Calculate consensus proportion
  consensus_proportion <- most_common_count / total_responses
  
  # Calculate Shannon entropy
  proportions <- as.numeric(response_counts) / total_responses
  entropy <- -sum(proportions * log2(proportions))
  
  
  return(list(
    consensus_proportion = consensus_proportion,
    entropy = entropy,
    majority_prediction = majority_prediction
  ))
}

# Constants for consensus checking
.CONSENSUS_CONSTANTS <- list(
  MAX_RETRIES = 3,
  DEFAULT_RESPONSE = "0\n0\n0\nUnknown",
  FALLBACK_MODELS = c("claude-sonnet-4.5", "gpt-5.2", "gemini-3-flash", "qwen3-max", "deepseek-r1"),
  NUMERIC_PATTERNS = list(
    CONSENSUS_INDICATOR = "^\\s*[01]\\s*$",
    PROPORTION = "^\\s*(0\\.\\d+|1\\.0*|1)\\s*$",
    ENTROPY = "^\\s*(\\d+\\.\\d+|\\d+)\\s*$",
    GENERAL_NUMERIC = "^\\s*\\d+(\\.\\d+)?\\s*$"
  ),
  SEARCH_PATTERNS = list(
    CONSENSUS_LABEL = "(C|c)onsensus (P|p)roportion",
    ENTROPY_LABEL = "(E|e)ntropy"
  )
)

# Default values for failed parsing
.DEFAULT_CONSENSUS_RESULT <- list(
  reached = FALSE,
  consensus_proportion = 0,
  entropy = 0,
  majority_prediction = "Unknown"
)

#' Prepare list of models to try for consensus checking
#
#
#' @keywords internal
prepare_models_list <- function(consensus_check_model = NULL) {
  models_to_try <- c()
  
  if (!is.null(consensus_check_model)) {
    get_logger()$info("Using specified consensus check model", list(model = consensus_check_model))
    
    if (grepl("/", consensus_check_model)) {
      parts <- strsplit(consensus_check_model, "/")[[1]]
      if (length(parts) > 1) {
        base_model <- parts[2]
        get_logger()$info("Detected OpenRouter model. Using base model name", list(base_model = base_model))
        models_to_try <- c(consensus_check_model, base_model)
      } else {
        models_to_try <- c(consensus_check_model)
      }
    } else {
      models_to_try <- c(consensus_check_model)
    }
  }
  
  fallback_models <- .CONSENSUS_CONSTANTS$FALLBACK_MODELS[!.CONSENSUS_CONSTANTS$FALLBACK_MODELS %in% models_to_try]
  c(models_to_try, fallback_models)
}

#' Parse standard 4-line consensus response format
#
#
#' @keywords internal
parse_standard_format <- function(result_lines) {
  if (length(result_lines) != 4) return(NULL)
  
  patterns <- .CONSENSUS_CONSTANTS$NUMERIC_PATTERNS
  is_line1_valid <- grepl(patterns$CONSENSUS_INDICATOR, result_lines[1])
  is_line2_valid <- grepl(patterns$PROPORTION, result_lines[2])
  is_line3_valid <- grepl(patterns$ENTROPY, result_lines[3])
  
  if (!all(c(is_line1_valid, is_line2_valid, is_line3_valid))) {
    return(NULL)
  }
  
  get_logger()$debug("Detected standard 4-line format")
  
  list(
    consensus = as.numeric(trimws(result_lines[1])) == 1,
    consensus_proportion = as.numeric(trimws(result_lines[2])),
    entropy = as.numeric(trimws(result_lines[3])),
    majority_prediction = trimws(result_lines[4])
  )
}

#' Extract numeric value from line containing a label
#
#
#
#
#' @keywords internal
extract_labeled_value <- function(lines, pattern, value_pattern) {
  for (line in lines) {
    if (grepl(pattern, line) && grepl("=", line)) {
      parts <- strsplit(line, "=")[[1]]
      if (length(parts) > 1) {
        last_part <- trimws(parts[length(parts)])
        num_match <- regexpr(value_pattern, last_part)
        if (num_match > 0) {
          value_str <- substr(last_part, num_match, num_match + attr(num_match, "match.length") - 1)
          value <- as.numeric(value_str)
          if (!is.na(value)) {
            get_logger()$debug("Found value in line", list(value = value, line = line))
            return(value)
          }
        }
      }
    }
  }
  NULL
}

#' Find majority prediction from response lines
#
#
#' @keywords internal
find_majority_prediction <- function(lines) {
  numeric_pattern <- .CONSENSUS_CONSTANTS$NUMERIC_PATTERNS$GENERAL_NUMERIC
  
  for (line in lines) {
    line_clean <- trimws(line)
    if (nchar(line_clean) == 0) next
    
    # Skip lines that match numeric patterns or contain labels
    if (grepl(numeric_pattern, line_clean) ||
        grepl(.CONSENSUS_CONSTANTS$NUMERIC_PATTERNS$CONSENSUS_INDICATOR, line_clean) ||
        grepl(.CONSENSUS_CONSTANTS$NUMERIC_PATTERNS$PROPORTION, line_clean) ||
        grepl(.CONSENSUS_CONSTANTS$NUMERIC_PATTERNS$ENTROPY, line_clean) ||
        grepl(.CONSENSUS_CONSTANTS$SEARCH_PATTERNS$CONSENSUS_LABEL, line_clean) ||
        grepl(.CONSENSUS_CONSTANTS$SEARCH_PATTERNS$ENTROPY_LABEL, line_clean)) {
      next
    }
    
    return(line_clean)
  }
  
  "Parsing_Failed"
}

#' Parse flexible format consensus response
#
#
#' @keywords internal
parse_flexible_format <- function(lines) {
  result <- list(
    consensus = FALSE,
    consensus_proportion = 0,
    entropy = 0,
    majority_prediction = "Unknown"
  )
  
  # Extract consensus indicator (0 or 1)
  for (line in lines) {
    if (grepl(.CONSENSUS_CONSTANTS$NUMERIC_PATTERNS$CONSENSUS_INDICATOR, line)) {
      result$consensus <- as.numeric(trimws(line)) == 1
      break
    }
  }
  
  # Extract consensus proportion
  proportion_value <- extract_labeled_value(
    lines, 
    .CONSENSUS_CONSTANTS$SEARCH_PATTERNS$CONSENSUS_LABEL, 
    "0\\.\\d+|1\\.0*|1"
  )
  if (!is.null(proportion_value) && proportion_value >= 0 && proportion_value <= 1) {
    result$consensus_proportion <- proportion_value
  }
  
  # Extract entropy
  entropy_value <- extract_labeled_value(
    lines, 
    .CONSENSUS_CONSTANTS$SEARCH_PATTERNS$ENTROPY_LABEL, 
    "\\d+\\.\\d+|\\d+"
  )
  if (!is.null(entropy_value) && entropy_value >= 0) {
    result$entropy <- entropy_value
  }
  
  # Extract majority prediction
  result$majority_prediction <- find_majority_prediction(lines)
  
  result
}

#' Parse consensus response from model
#
#
#' @keywords internal
parse_consensus_response <- function(response) {
  # Handle NULL or empty response
  if (is.null(response) || length(response) == 0) {
    get_logger()$warn("Response is NULL, empty, or has zero length")
    return(.DEFAULT_CONSENSUS_RESULT)
  }
  
  # Handle non-character responses
  if (!is.character(response)) {
    get_logger()$warn("Response is not character, converting", list(type = typeof(response)))
    if (is.function(response)) {
      get_logger()$error("Response is a function, indicating serious API error")
      return(.DEFAULT_CONSENSUS_RESULT)
    }
    
    response <- tryCatch({
      as.character(response)
    }, error = function(e) {
      get_logger()$error("Failed to convert response", list(error = e$message))
      return(NULL)
    })
    if (is.null(response)) return(.DEFAULT_CONSENSUS_RESULT)
  }

  if (length(response) > 1) {
    response <- paste(response, collapse = "\n")
  }
  
  # Check for empty/NA string after conversion
  if (length(response) == 0 || is.na(response) || !nzchar(response)) {
    get_logger()$warn("Response is empty string")
    return(.DEFAULT_CONSENSUS_RESULT)
  }
  
  # Split response into lines
  lines <- if (grepl("\n", response)) {
    tryCatch({
      split_lines <- strsplit(response, "\n")[[1]]
      trimws(split_lines[nchar(split_lines) > 0])
    }, error = function(e) {
      get_logger()$error("Failed to split response", list(error = e$message))
      c(response)
    })
  } else {
    c(response)
  }
  
  if (length(lines) < 4) {
    get_logger()$warn("Not enough lines for standard format, trying flexible parsing", list(line_count = length(lines)))
  } else {
    # Try standard format first
    result_lines <- tail(lines, 4)
    standard_result <- parse_standard_format(result_lines)
    
    if (!is.null(standard_result)) {
      get_logger()$info("Parsed standard format", list(
        consensus = standard_result$consensus,
        proportion = standard_result$consensus_proportion,
        entropy = standard_result$entropy
      ))
      return(list(
        reached = standard_result$consensus,
        consensus_proportion = standard_result$consensus_proportion,
        entropy = standard_result$entropy,
        majority_prediction = standard_result$majority_prediction
      ))
    }
  }
  
  # Fall back to flexible parsing
  get_logger()$debug("Using flexible format parsing")
  flexible_result <- parse_flexible_format(lines)
  
  list(
    reached = flexible_result$consensus,
    consensus_proportion = flexible_result$consensus_proportion,
    entropy = flexible_result$entropy,
    majority_prediction = flexible_result$majority_prediction
  )
}

#' Execute consensus check with retry logic
#
#
#
#
#' @keywords internal
execute_consensus_check <- function(formatted_responses, api_keys, models_to_try, base_urls = NULL) {
  max_retries <- .CONSENSUS_CONSTANTS$MAX_RETRIES
  
  for (model_name in models_to_try) {
    get_logger()$info("Trying model for consensus check", list(model = model_name))
    
    # Get API key
    api_key <- get_api_key(model_name, api_keys)
    if (is.null(api_key) || nchar(api_key) == 0) {
      provider <- tryCatch({
        get_provider(model_name)
      }, error = function(e) {
        get_logger()$error("Could not determine provider for model", list(model = model_name, error = e$message))
        return(NULL)
      })
      
      if (!is.null(provider)) {
        env_var <- paste0(toupper(provider), "_API_KEY")
        api_key <- Sys.getenv(env_var)
      }
    }
    
    if (is.null(api_key) || nchar(api_key) == 0) {
      get_logger()$warn("No API key available for model, skipping", list(model = model_name))
      next
    }
    
    # Retry logic for current model
    for (attempt in 1:max_retries) {
      get_logger()$debug("Starting attempt with model", list(attempt = attempt, max_retries = max_retries, model = model_name))
      
      result <- tryCatch({
        temp_response <- get_model_response(formatted_responses, model_name, api_key, base_urls)
        
        if (is.character(temp_response) && length(temp_response) > 1) {
          temp_response <- paste(temp_response, collapse = "\n")
        }
        
        get_logger()$info("Successfully got response from model", list(model = model_name))
        list(success = TRUE, response = temp_response)
      }, error = function(e) {
        get_logger()$error("API call failed on attempt", list(model = model_name, attempt = attempt, error = e$message))
        
        if (attempt < max_retries) {
          wait_time <- 5 * 2^(attempt-1)
          get_logger()$info("Waiting before next attempt", list(wait_seconds = wait_time))
          Sys.sleep(wait_time)
        }
        
        list(success = FALSE, response = NULL)
      })
      
      if (result$success) {
        return(result)
      }
    }
  }
  
  # All attempts failed
  get_logger()$warn("All model attempts failed, consensus check could not be performed")
  warning("All available models failed for consensus check. Please ensure at least one model API key is valid.")
  list(success = FALSE, response = .CONSENSUS_CONSTANTS$DEFAULT_RESPONSE)
}

#' Check if consensus is reached among models
#
#
#
#
#
#' @note This function uses create_consensus_check_prompt from prompt_templates.R
#' @importFrom utils tail
#' @keywords internal
check_consensus <- function(round_responses, api_keys = NULL, controversy_threshold = 2/3, entropy_threshold = 1.0, consensus_check_model = NULL, base_urls = NULL) {
  # Initialize logging
  get_logger()$info("Starting check_consensus function")
  get_logger()$debug("Input responses", list(responses = round_responses))

  # Validate input
  if (length(round_responses) < 2) {
    get_logger()$warn("Not enough responses to check consensus", list(response_count = length(round_responses)))
    return(list(reached = FALSE, consensus_proportion = 0, entropy = 0, majority_prediction = "Insufficient_Responses"))
  }

  # OPTIMIZATION: First try simple consensus calculation
  get_logger()$info("Starting with simple consensus calculation")
  
  # Extract cell types from responses if they are discussion format
  extracted_cell_types <- sapply(round_responses, function(response) {
    if (is.character(response) && length(response) > 1) {
      # Multi-line response (discussion format)
      # Look for line starting with "CELL TYPE:"
      cell_type_lines <- grep("^CELL TYPE:", response, value = TRUE, ignore.case = TRUE)
      if (length(cell_type_lines) > 0) {
        # Extract cell type after "CELL TYPE:"
        cell_type <- trimws(sub("^CELL TYPE:\\s*", "", cell_type_lines[1], ignore.case = TRUE))
        return(cell_type)
      }
    } else if (is.character(response) && length(response) == 1) {
      # Single line response - return as is
      return(response)
    }
    # Default to "Unknown" if extraction fails
    return("Unknown")
  })
  
  
  # Calculate simple consensus
  simple_result <- calculate_simple_consensus(extracted_cell_types)
  
  
  # Check if simple consensus meets thresholds
  if (simple_result$consensus_proportion >= controversy_threshold && 
      simple_result$entropy <= entropy_threshold) {
    # Simple consensus is sufficient
    get_logger()$info("CONSENSUS ACHIEVED WITH SIMPLE CHECK - NO LLM NEEDED", list(
      proportion = simple_result$consensus_proportion,
      entropy = simple_result$entropy,
      controversy_threshold = controversy_threshold,
      entropy_threshold = entropy_threshold,
      majority = simple_result$majority_prediction
    ))
    
    message(sprintf("Cluster consensus achieved via SIMPLE CHECK: %s (CP=%.2f, H=%.2f)",
                    simple_result$majority_prediction,
                    simple_result$consensus_proportion,
                    simple_result$entropy))
    
    return(list(
      reached = TRUE,
      consensus_proportion = simple_result$consensus_proportion,
      entropy = simple_result$entropy,
      majority_prediction = simple_result$majority_prediction
    ))
  }
  
  # Simple consensus didn't meet thresholds, use LLM for double-checking
  get_logger()$info("Simple consensus BELOW threshold, REQUIRING LLM double-check", list(
    proportion = simple_result$consensus_proportion,
    entropy = simple_result$entropy,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    reason = if(simple_result$consensus_proportion < controversy_threshold) 
             "Low consensus proportion" else "High entropy"
  ))
  
  message(sprintf("Simple check insufficient (CP=%.2f < %.2f OR H=%.2f > %.2f) - Using LLM",
                  simple_result$consensus_proportion,
                  controversy_threshold,
                  simple_result$entropy,
                  entropy_threshold))

  # Get the formatted prompt from the dedicated function
  formatted_responses <- create_consensus_check_prompt(round_responses, controversy_threshold, entropy_threshold)

  # Prepare models and execute consensus check
  models_to_try <- prepare_models_list(consensus_check_model)
  execution_result <- execute_consensus_check(formatted_responses, api_keys, models_to_try, base_urls)

  # Handle execution failure - fall back to simple consensus
  if (!execution_result$success) {
    get_logger()$error("All model attempts failed, using simple consensus results")
    return(list(
      reached = simple_result$consensus_proportion >= controversy_threshold && 
               simple_result$entropy <= entropy_threshold,
      consensus_proportion = simple_result$consensus_proportion,
      entropy = simple_result$entropy,
      majority_prediction = simple_result$majority_prediction
    ))
  }

  # Parse the response using the new modular approach
  result <- parse_consensus_response(execution_result$response)
  
  
  # Compare LLM result with simple consensus
  if (result$reached) {
    message(sprintf("LLM CONFIRMED consensus: %s (CP=%.2f, H=%.2f)",
                    result$majority_prediction,
                    result$consensus_proportion,
                    result$entropy))
  } else {
    message(sprintf("LLM REJECTED consensus: Simple suggested %s but LLM found CP=%.2f, H=%.2f",
                    simple_result$majority_prediction,
                    result$consensus_proportion,
                    result$entropy))
  }
  

  return(result)
}
