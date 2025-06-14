# Constants for consensus checking
.CONSENSUS_CONSTANTS <- list(
  MAX_RETRIES = 3,
  DEFAULT_RESPONSE = "0\n0\n0\nUnknown",
  FALLBACK_MODELS = c("qwen-max-2025-01-25", "claude-3-5-sonnet-20241022", "gpt-4o", "gemini-2.0-flash"),
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
#' @param consensus_check_model User-specified model (can be NULL)
#' @return Character vector of models in order of preference
prepare_models_list <- function(consensus_check_model = NULL) {
  models_to_try <- c()
  
  if (!is.null(consensus_check_model)) {
    write_log(sprintf("Using specified consensus check model: %s", consensus_check_model))
    
    if (grepl("/", consensus_check_model)) {
      parts <- strsplit(consensus_check_model, "/")[[1]]
      if (length(parts) > 1) {
        base_model <- parts[2]
        write_log(sprintf("Detected OpenRouter model. Using base model name: %s", base_model))
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
#' @param result_lines Character vector of 4 lines
#' @return List with parsed values or NULL if not standard format
parse_standard_format <- function(result_lines) {
  if (length(result_lines) != 4) return(NULL)
  
  patterns <- .CONSENSUS_CONSTANTS$NUMERIC_PATTERNS
  is_line1_valid <- grepl(patterns$CONSENSUS_INDICATOR, result_lines[1])
  is_line2_valid <- grepl(patterns$PROPORTION, result_lines[2])
  is_line3_valid <- grepl(patterns$ENTROPY, result_lines[3])
  
  if (!all(c(is_line1_valid, is_line2_valid, is_line3_valid))) {
    return(NULL)
  }
  
  write_log("Detected standard 4-line format")
  
  list(
    consensus = as.numeric(trimws(result_lines[1])) == 1,
    consensus_proportion = as.numeric(trimws(result_lines[2])),
    entropy = as.numeric(trimws(result_lines[3])),
    majority_prediction = trimws(result_lines[4])
  )
}

#' Extract numeric value from line containing a label
#' @param lines Character vector of all response lines
#' @param pattern Pattern to match the label
#' @param value_pattern Pattern to extract the numeric value
#' @return Numeric value or NULL if not found
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
            write_log(sprintf("Found value %f in line: %s", value, line))
            return(value)
          }
        }
      }
    }
  }
  NULL
}

#' Find majority prediction from response lines
#' @param lines Character vector of response lines
#' @return Character string of majority prediction
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
#' @param lines Character vector of all response lines
#' @return List with parsed values
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
#' @param response Character string response from model
#' @return List with consensus results
parse_consensus_response <- function(response) {
  # Handle NULL or empty response
  if (is.null(response) || length(response) == 0) {
    write_log("WARNING: Response is NULL, empty, or has zero length")
    return(.DEFAULT_CONSENSUS_RESULT)
  }
  
  # Handle non-character responses
  if (!is.character(response)) {
    write_log(sprintf("WARNING: Response is not character but %s, converting", typeof(response)))
    if (is.function(response)) {
      write_log("ERROR: Response is a function, indicating serious API error")
      return(.DEFAULT_CONSENSUS_RESULT)
    }
    
    tryCatch({
      response <- as.character(response)
    }, error = function(e) {
      write_log(sprintf("ERROR: Failed to convert response: %s", e$message))
      return(.DEFAULT_CONSENSUS_RESULT)
    })
  }
  
  # Check for empty string after conversion
  if (nchar(response) == 0) {
    write_log("WARNING: Response is empty string")
    return(.DEFAULT_CONSENSUS_RESULT)
  }
  
  # Split response into lines
  lines <- if (grepl("\n", response)) {
    tryCatch({
      split_lines <- strsplit(response, "\n")[[1]]
      trimws(split_lines[nchar(split_lines) > 0])
    }, error = function(e) {
      write_log(sprintf("ERROR: Failed to split response: %s", e$message))
      return(c(response))
    })
  } else {
    c(response)
  }
  
  if (length(lines) < 4) {
    write_log("WARNING: Not enough lines in response")
    return(.DEFAULT_CONSENSUS_RESULT)
  }
  
  # Try standard format first
  result_lines <- tail(lines, 4)
  standard_result <- parse_standard_format(result_lines)
  
  if (!is.null(standard_result)) {
    write_log(sprintf("Parsed standard format: consensus=%s, proportion=%f, entropy=%f",
                     standard_result$consensus, standard_result$consensus_proportion, standard_result$entropy))
    return(list(
      reached = standard_result$consensus,
      consensus_proportion = standard_result$consensus_proportion,
      entropy = standard_result$entropy,
      majority_prediction = standard_result$majority_prediction
    ))
  }
  
  # Fall back to flexible parsing
  write_log("Using flexible format parsing")
  flexible_result <- parse_flexible_format(lines)
  
  list(
    reached = flexible_result$consensus,
    consensus_proportion = flexible_result$consensus_proportion,
    entropy = flexible_result$entropy,
    majority_prediction = flexible_result$majority_prediction
  )
}

#' Execute consensus check with retry logic
#' @param formatted_responses Formatted prompt for consensus check
#' @param api_keys List of API keys
#' @param models_to_try Character vector of models to attempt
#' @return List with success flag and response
execute_consensus_check <- function(formatted_responses, api_keys, models_to_try) {
  max_retries <- .CONSENSUS_CONSTANTS$MAX_RETRIES
  
  for (model_name in models_to_try) {
    write_log(sprintf("Trying model %s for consensus check", model_name))
    
    # Get API key
    api_key <- get_api_key(model_name, api_keys)
    if (is.null(api_key) || nchar(api_key) == 0) {
      provider <- tryCatch({
        get_provider(model_name)
      }, error = function(e) {
        write_log(sprintf("ERROR: Could not determine provider for %s: %s", model_name, e$message))
        return(NULL)
      })
      
      if (!is.null(provider)) {
        env_var <- paste0(toupper(provider), "_API_KEY")
        api_key <- Sys.getenv(env_var)
      }
    }
    
    if (is.null(api_key) || nchar(api_key) == 0) {
      write_log(sprintf("No API key available for %s, skipping", model_name))
      next
    }
    
    # Retry logic for current model
    for (attempt in 1:max_retries) {
      write_log(sprintf("Attempt %d of %d with model %s", attempt, max_retries, model_name))
      
      result <- tryCatch({
        temp_response <- get_model_response(formatted_responses, model_name, api_key)
        
        if (is.character(temp_response) && length(temp_response) > 1) {
          temp_response <- paste(temp_response, collapse = "\n")
        }
        
        write_log(sprintf("Successfully got response from %s", model_name))
        list(success = TRUE, response = temp_response)
      }, error = function(e) {
        write_log(sprintf("ERROR on %s attempt %d: %s", model_name, attempt, e$message))
        
        if (attempt < max_retries) {
          wait_time <- 5 * 2^(attempt-1)
          write_log(sprintf("Waiting for %d seconds before next attempt...", wait_time))
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
  write_log("WARNING: All model attempts failed, consensus check could not be performed")
  warning("All available models failed for consensus check. Please ensure at least one model API key is valid.")
  list(success = FALSE, response = .CONSENSUS_CONSTANTS$DEFAULT_RESPONSE)
}

#' Check if consensus is reached among models
#' @param round_responses A vector of model responses to check for consensus
#' @param api_keys A list of API keys for different providers
#' @param controversy_threshold Threshold for consensus proportion (default: 2/3)
#' @param entropy_threshold Threshold for entropy (default: 1.0)
#' @param consensus_check_model Model to use for consensus checking (default: NULL, will try available models in order)
#' @note This function uses create_consensus_check_prompt from prompt_templates.R
#' @importFrom utils write.table tail
#' @keywords internal
check_consensus <- function(round_responses, api_keys = NULL, controversy_threshold = 2/3, entropy_threshold = 1.0, consensus_check_model = NULL) {
  # Initialize logging
  write_log("\n=== Starting check_consensus function ===")
  write_log(sprintf("Input responses: %s", paste(round_responses, collapse = "; ")))

  # Validate input
  if (length(round_responses) < 2) {
    write_log("WARNING: Not enough responses to check consensus")
    return(list(reached = FALSE, consensus_proportion = 0, entropy = 0, majority_prediction = "Insufficient_Responses"))
  }

  # Get the formatted prompt from the dedicated function
  formatted_responses <- create_consensus_check_prompt(round_responses, controversy_threshold, entropy_threshold)

  # Prepare models and execute consensus check
  models_to_try <- prepare_models_list(consensus_check_model)
  execution_result <- execute_consensus_check(formatted_responses, api_keys, models_to_try)

  # Handle execution failure
  if (!execution_result$success) {
    write_log("All model attempts failed, using default values")
    return(.DEFAULT_CONSENSUS_RESULT)
  }

  # Parse the response using the new modular approach
  result <- parse_consensus_response(execution_result$response)
  
  # Log final results
  write_log(sprintf("Final results: consensus=%s, proportion=%f, entropy=%f, majority=%s",
                   ifelse(result$reached, "TRUE", "FALSE"),
                   result$consensus_proportion,
                   result$entropy,
                   result$majority_prediction))

  return(result)
}
