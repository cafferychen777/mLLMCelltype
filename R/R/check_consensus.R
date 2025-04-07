#' Check if consensus is reached among models using Claude
#' @param round_responses A vector of model responses to check for consensus
#' @param api_keys A list of API keys for different providers
#' @note This function uses create_consensus_check_prompt from prompt_templates.R
#' @importFrom utils write.table tail
#' @keywords internal
check_consensus <- function(round_responses, api_keys = NULL) {
  # Initialize logging
  write_log("\n=== Starting check_consensus function ===")
  write_log(sprintf("Input responses: %s", paste(round_responses, collapse = "; ")))
  
  # Validate input
  if (length(round_responses) < 2) {
    write_log("WARNING: Not enough responses to check consensus")
    return(list(reached = FALSE, consensus_proportion = 0, entropy = 0, majority_prediction = NA))
  }
  
  # Get the formatted prompt from the dedicated function
  formatted_responses <- create_consensus_check_prompt(round_responses)
  
  # Get model analysis with retry mechanism
  max_retries <- 3
  response <- "0\n0\n0\nUnknown"  # Default response in case all attempts fail
  
  # First try with Qwen
  for (attempt in 1:max_retries) {
    write_log(sprintf("Attempt %d of %d to get Qwen's analysis", attempt, max_retries))
    tryCatch({
      # Get API key using the get_api_key function
      qwen_api_key <- get_api_key("qwen-max-2025-01-25", api_keys)
      if (is.null(qwen_api_key)) {
        qwen_api_key <- Sys.getenv("QWEN_API_KEY")
      }
      
      response <- process_qwen(
        formatted_responses,
        "qwen-max-2025-01-25",
        qwen_api_key
      )
      # Ensure response is a single string
      if (is.character(response) && length(response) > 1) {
        response <- paste(response, collapse = "\n")
      }
      write_log(sprintf("Successfully got response from Qwen on attempt %d", attempt))
      break
    }, error = function(e) {
      write_log(sprintf("ERROR on Qwen attempt %d: %s", attempt, e$message))
      if (attempt == max_retries) {
        write_log("All Qwen retry attempts failed, falling back to Claude")
        # Try Claude as fallback
        tryCatch({
          # Get API key using the get_api_key function
          anthropic_api_key <- get_api_key("claude-3-5-sonnet-latest", api_keys)
          if (is.null(anthropic_api_key)) {
            anthropic_api_key <- Sys.getenv("ANTHROPIC_API_KEY")
          }
          
          response <- process_anthropic(
            formatted_responses,
            "claude-3-5-sonnet-latest",
            anthropic_api_key
          )
          # Ensure response is a single string
          if (is.character(response) && length(response) > 1) {
            response <- paste(response, collapse = "\n")
          }
          write_log("Successfully got response from Claude as fallback")
        }, error = function(e) {
          write_log(sprintf("ERROR on Claude fallback: %s", e$message))
          warning(sprintf("Both Qwen and Claude failed: %s", e$message))
        })
      } else {
        message(sprintf("Qwen attempt %d failed: %s. Retrying...", attempt, e$message))
        wait_time <- 5 * 2^(attempt-1)  
        message(sprintf("Waiting for %d seconds before next attempt...", wait_time))
        Sys.sleep(wait_time)
      }
    })
  }
  
  # Directly parse the response using a simpler approach
  # First, check if the response contains newlines
  if (grepl("\n", response)) {
    # Split by newlines and clean up
    lines <- strsplit(response, "\n")[[1]]
    lines <- trimws(lines)
    lines <- lines[nchar(lines) > 0]
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
        
        for (i in 1:length(lines)) {
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
        
        for (i in 1:length(lines)) {
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
      for (i in 1:length(lines)) {
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
      majority_prediction <- "Unknown"
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
