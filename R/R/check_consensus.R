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
  # Parameters are used in prompt template to instruct LLM on threshold values # nolint
  formatted_responses <- create_consensus_check_prompt(round_responses, controversy_threshold, entropy_threshold)

  # Get model analysis with retry mechanism
  max_retries <- 3
  response <- "0\n0\n0\nUnknown"  # Default response in case all attempts fail
  success <- FALSE

  # Define models to try, in order of preference
  models_to_try <- c()

  # If consensus_check_model is specified, prioritize it
  if (!is.null(consensus_check_model)) {
    write_log(sprintf("Using specified consensus check model: %s", consensus_check_model))

    # Check if this is an OpenRouter model (contains a slash)
    if (grepl("/", consensus_check_model)) {
      # For OpenRouter models, we need to extract the base model name
      # Format is typically "provider/model" like "google/gemini-2.5-pro-preview-03-25"
      parts <- strsplit(consensus_check_model, "/")[[1]]
      if (length(parts) > 1) {
        # Use the model part (after the slash)
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

  # Add fallback models
  fallback_models <- c("qwen-max-2025-01-25", "claude-3-5-sonnet-20241022", "gpt-4o", "gemini-2.0-flash")
  # Remove any models that are already in models_to_try
  fallback_models <- fallback_models[!fallback_models %in% models_to_try]
  models_to_try <- c(models_to_try, fallback_models)

  # Try each model in order
  for (model_name in models_to_try) {
    if (success) break

    write_log(sprintf("Trying model %s for consensus check", model_name))

    # Get API key for this model
    api_key <- get_api_key(model_name, api_keys)
    if (is.null(api_key) || nchar(api_key) == 0) {
      # Try to get from environment variables based on provider
      provider <- tryCatch({
        get_provider(model_name)
      }, error = function(e) {
        write_log(sprintf("ERROR: Could not determine provider for model %s: %s", model_name, e$message))
        return(NULL)
      })

      if (!is.null(provider)) {
        env_var <- paste0(toupper(provider), "_API_KEY")
        api_key <- Sys.getenv(env_var)
      }
    }

    # Skip if no API key available
    if (is.null(api_key) || nchar(api_key) == 0) {
      write_log(sprintf("No API key available for %s, skipping", model_name))
      next
    }

    # Try with current model using retry mechanism
    for (attempt in 1:max_retries) {
      write_log(sprintf("Attempt %d of %d with model %s", attempt, max_retries, model_name))

      tryCatch({
        # Use get_model_response which automatically selects the right processor
        temp_response <- get_model_response(
          formatted_responses,
          model_name,
          api_key
        )

        # Ensure response is a single string
        if (is.character(temp_response) && length(temp_response) > 1) {
          temp_response <- paste(temp_response, collapse = "\n")
        }

        write_log(sprintf("Successfully got response from %s", model_name))
        response <- temp_response
        success <- TRUE
        break
      }, error = function(e) {
        write_log(sprintf("ERROR on %s attempt %d: %s", model_name, attempt, e$message))

        if (attempt < max_retries) {
          wait_time <- 5 * 2^(attempt-1)
          write_log(sprintf("Waiting for %d seconds before next attempt...", wait_time))
          Sys.sleep(wait_time)
        }
      })

      if (success) break
    }
  }

  # If all models failed, return default values with warning
  if (!success) {
    # Note: We don't use a local statistical method here because simple statistical methods
    # (like majority voting or frequency counting) are too simplistic for cell type annotation.
    # They cannot capture semantic similarities between different annotations and would lead to
    # poor quality results. Cell type annotation requires understanding of biological context and
    # semantic relationships, which only LLMs can provide effectively.
    write_log("WARNING: All model attempts failed, consensus check could not be performed")
    warning("All available models failed for consensus check. Please ensure at least one model API key is valid.")
    return(list(reached = FALSE, consensus_proportion = 0, entropy = 0, majority_prediction = "Unknown"))
  }

  # Directly parse the response using a simpler approach
  # First, check if response is NULL or empty
  if (is.null(response) || length(response) == 0) {
    write_log("WARNING: Response is NULL or has zero length")
    lines <- c("0", "0", "0", "Unknown")
  } else if (!is.character(response)) {
    # If response is not a character, convert it to string
    write_log(sprintf("WARNING: Response is not a character but %s, converting to string", typeof(response)))
    # Handle different types more carefully
    if (is.function(response)) {
      write_log("ERROR: Response is a function (closure), this indicates a serious error in the API response processing")
      lines <- c("0", "0", "0", "Unknown")
    } else {
      tryCatch({
        response <- as.character(response)
        if (nchar(response) == 0) {
          lines <- c("0", "0", "0", "Unknown")
        } else {
          lines <- c(response)
        }
      }, error = function(e) {
        write_log(sprintf("ERROR: Failed to convert response to character: %s", e$message))
        lines <- c("0", "0", "0", "Unknown")
      })
    }
  } else if (nchar(response) == 0) {
    write_log("WARNING: Response is empty string")
    lines <- c("0", "0", "0", "Unknown")
  } else if (grepl("\n", response)) {
    # Split by newlines and clean up
    tryCatch({
      lines <- strsplit(response, "\n")[[1]]
      lines <- trimws(lines)
      lines <- lines[nchar(lines) > 0]
    }, error = function(e) {
      write_log(sprintf("ERROR: Failed to split response by newlines: %s", e$message))
      lines <- c("0", "0", "0", "Unknown")
    })
  } else {
    # If no newlines, treat as a single line
    lines <- c(response)
  }

  # Get the last 4 non-empty lines, as the model might include explanatory text before the actual results
  if (length(lines) >= 4) {
    # Extract the last 4 lines
    result_lines <- tail(lines, 4)

    # First try to process the standard four-line format
    # Check if it's a standard four-line format
    standard_format <- FALSE
    if (length(result_lines) == 4) {
      # Check if the first line is 0 or 1
      is_line1_valid <- grepl("^\\s*[01]\\s*$", result_lines[1])
      # Check if the second line is a decimal between 0-1
      is_line2_valid <- grepl("^\\s*(0\\.\\d+|1\\.0*|1)\\s*$", result_lines[2])
      # Check if the third line is a non-negative number
      is_line3_valid <- grepl("^\\s*(\\d+\\.\\d+|\\d+)\\s*$", result_lines[3])

      if (is_line1_valid && is_line2_valid && is_line3_valid) {
        standard_format <- TRUE
        write_log("Detected standard 4-line format")

        # Extract consensus indicator
        consensus_value <- as.numeric(trimws(result_lines[1]))
        consensus <- consensus_value == 1

        # Extract consensus proportion
        consensus_proportion <- as.numeric(trimws(result_lines[2]))
        proportion_found <- TRUE
        write_log(sprintf("Found proportion value %f in standard format line 2", consensus_proportion))

        # Extract entropy value
        entropy <- as.numeric(trimws(result_lines[3]))
        entropy_found <- TRUE
        write_log(sprintf("Found entropy value %f in standard format line 3", entropy))

        # Extract majority prediction result
        majority_prediction <- trimws(result_lines[4])
      }
    }

    # Only execute complex parsing logic when not in standard format
    if (!standard_format) {
      # Try to find the most likely numeric values in the last 4 lines
      # Look for lines that start with a number or are just a number
      numeric_pattern <- "^\\s*([01]|0\\.[0-9]+|1\\.[0-9]+)\\s*$"

      # Check if the first line is a valid consensus indicator (0 or 1)
      if (grepl("^\\s*[01]\\s*$", result_lines[1])) {
        consensus_value <- as.numeric(trimws(result_lines[1]))
        consensus <- consensus_value == 1
      } else {
        # Try to find a 0 or 1 in any of the lines
        for (i in 1:4) {
          if (grepl("^\\s*[01]\\s*$", result_lines[i])) {
            consensus_value <- as.numeric(trimws(result_lines[i]))
            consensus <- consensus_value == 1
            # Reorder the lines if needed
            if (i > 1) {
              result_lines <- c(result_lines[i], result_lines[-i])
            }
            break
          }
        }
        if (!exists("consensus")) {
          write_log("WARNING: Could not find valid consensus indicator (0 or 1)")
          consensus <- FALSE
        }
      }

      # Look for a proportion value (between 0 and 1)
      if (!exists("proportion_found") || !proportion_found) {
        proportion_found <- FALSE

        for (i in seq_along(lines)) {
          if (grepl("(C|c)onsensus (P|p)roportion", lines[i]) && grepl("=", lines[i])) {
            # Extract all content after the equals sign
            parts <- strsplit(lines[i], "=")[[1]]
            if (length(parts) > 1) {
              # Get the last part (may contain multiple equals signs)
              last_part <- trimws(parts[length(parts)])
              # Try to extract the number
              num_match <- regexpr("0\\.\\d+|1\\.0*|1", last_part)
              if (num_match > 0) {
                value_str <- substr(last_part, num_match, num_match + attr(num_match, "match.length") - 1)
                value <- as.numeric(value_str)
                if (!is.na(value) && value >= 0 && value <= 1) {
                  consensus_proportion <- value
                  proportion_found <- TRUE
                  write_log(sprintf("Found proportion value %f in line: %s", value, lines[i]))
                  break
                }
              }
            }
          }
        }

        if (!proportion_found) {
          write_log("WARNING: Invalid consensus proportion, setting to 0")
          consensus_proportion <- 0
        }
      }

      # Look for an entropy value (a non-negative number, often with decimal places)
      if (!exists("entropy_found") || !entropy_found) {
        entropy_found <- FALSE

        for (i in seq_along(lines)) {
          if (grepl("(E|e)ntropy", lines[i]) && grepl("=", lines[i])) {
            # Extract all content after the equals sign
            parts <- strsplit(lines[i], "=")[[1]]
            if (length(parts) > 1) {
              # Get the last part (may contain multiple equals signs)
              last_part <- trimws(parts[length(parts)])
              # Try to extract the number
              num_match <- regexpr("\\d+\\.\\d+|\\d+", last_part)
              if (num_match > 0) {
                value_str <- substr(last_part, num_match, num_match + attr(num_match, "match.length") - 1)
                value <- as.numeric(value_str)
                if (!is.na(value) && value >= 0) {
                  entropy <- value
                  entropy_found <- TRUE
                  write_log(sprintf("Found entropy value %f in line: %s", value, lines[i]))
                  break
                }
              }
            }
          }
        }

        if (!entropy_found) {
          write_log("WARNING: Invalid entropy, setting to 0")
          entropy <- 0
        }
      }

      # Look for the majority prediction (a non-numeric line)
      numeric_pattern <- "^\\s*\\d+(\\.\\d+)?\\s*$"
      majority_prediction <- NULL

      # First try to extract from the last four lines
      for (i in 1:4) {
        if (!grepl(numeric_pattern, result_lines[i]) &&
            !grepl("0\\.\\d+|1\\.0*|1", result_lines[i]) &&
            !grepl("\\d+\\.\\d+|\\d+", result_lines[i])) {
          # This line doesn't match any numeric pattern, likely the cell type
          majority_prediction <- trimws(result_lines[i])
          break
        }
      }
    }

    # If we still don't have a majority prediction, search all lines
    if (is.null(majority_prediction)) {
      for (i in seq_along(lines)) {
        if (!grepl(numeric_pattern, lines[i]) &&
            !grepl("(C|c)onsensus", lines[i]) &&
            !grepl("(E|e)ntropy", lines[i]) &&
            !grepl("^\\s*[01]\\s*$", lines[i]) &&
            !grepl("^\\s*(0\\.\\d+|1\\.0*|1)\\s*$", lines[i]) &&
            !grepl("^\\s*(\\d+\\.\\d+|\\d+)\\s*$", lines[i]) &&
            nchar(trimws(lines[i])) > 0) {
          majority_prediction <- trimws(lines[i])
          break
        }
      }
    }

    if (is.null(majority_prediction)) {
      write_log("WARNING: Could not find a valid majority prediction")
      majority_prediction <- "Parsing_Failed"
    }
  } else {
    write_log("WARNING: Not enough lines in response")
    consensus <- FALSE
    consensus_proportion <- 0
    entropy <- 0
    majority_prediction <- "Unknown"
  }

  # Return the results
  write_log(sprintf("Final results: consensus=%s, proportion=%f, entropy=%f, majority=%s",
                   ifelse(consensus, "TRUE", "FALSE"),
                   consensus_proportion,
                   entropy,
                   majority_prediction))

  return(list(
    reached = consensus,
    consensus_proportion = consensus_proportion,
    entropy = entropy,
    majority_prediction = majority_prediction
  ))
}
