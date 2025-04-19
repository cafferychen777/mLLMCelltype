#' Process request using X.AI Grok models
#' @param prompt The prompt to send to the Grok API
#' @param model The Grok model to use
#' @param api_key The API key for X.AI Grok
#' @return The response from the Grok API
#' @importFrom httr POST add_headers content http_error http_status
#' @importFrom jsonlite toJSON fromJSON
#' @keywords internal
process_grok <- function(prompt, model, api_key) {
  write_log(sprintf("Starting X.AI Grok API request with model: %s", model))
  
  # X.AI Grok API endpoint
  url <- "https://api.x.ai/v1/chat/completions"
  write_log(sprintf("Using model: %s", model))
  
  # Process all input at once
  input_lines <- strsplit(prompt, "\n")[[1]]
  cutnum <- 1  # Always use 1 chunk for processing
  
  write_log(sprintf("Processing %d chunks of input", cutnum))
  
  if (cutnum > 1) {
    cid <- as.numeric(cut(seq_along(input_lines), cutnum))	
  } else {
    cid <- rep(1, length(input_lines))
  }
  
  # Process each chunk
  allres <- sapply(seq_len(cutnum), function(i) {
    write_log(sprintf("Processing chunk %d of %d", i, cutnum))
    id <- which(cid == i)
    
    # Prepare the request body
    body <- list(
      model = model,
      temperature = 0.7,
      stream = FALSE,
      messages = list(
        list(
          role = "user",
          content = paste(input_lines[id], collapse = '\n')
        )
      )
    )
    
    write_log("Sending API request...")
    # Make the API request
    response <- httr::POST(
      url = url,
      httr::add_headers(
        "Authorization" = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      ),
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      encode = "json"
    )
    
    # Check for errors
    if (!inherits(response, "response")) {
      write_log("ERROR: Invalid response object")
      return(NULL)
    }
    
    if (response$status_code >= 400) {
      error_message <- httr::content(response, "parsed")
      write_log(sprintf("ERROR: X.AI Grok API request failed: %s", 
                       if (!is.null(error_message$error$message)) error_message$error$message else "Unknown error"))
      return(NULL)
    }
    
    write_log("Parsing API response...")
    # Parse the response
    content <- httr::content(response, "parsed")
    
    # Extract the response text from the Grok API response
    # The standard format is:
    # {
    #   "id": "0daf962f-a275-4a3c-839a-047854645532",
    #   "object": "chat.completion",
    #   "created": 1739301120,
    #   "model": "grok-3-latest",
    #   "choices": [
    #     {
    #       "index": 0,
    #       "message": {
    #         "role": "assistant",
    #         "content": "...",
    #         "refusal": null
    #       },
    #       "finish_reason": "stop"
    #     }
    #   ],
    #   "usage": {...},
    #   "system_fingerprint": "fp_84ff176447"
    # }
    
    # Check if we have a valid response structure
    if (is.null(content$choices) || length(content$choices) == 0 || 
        is.null(content$choices[[1]]$message) || 
        is.null(content$choices[[1]]$message$content)) {
      write_log("ERROR: Unexpected response format from Grok API")
      write_log(sprintf("Raw response: %s", jsonlite::toJSON(content, auto_unbox = TRUE)))
      return(NULL)
    }
    
    # Extract the content from the message
    res <- strsplit(content$choices[[1]]$message$content, '\n')[[1]]
    
    # Log usage information if available
    if (!is.null(content$usage)) {
      write_log(sprintf("Tokens used - Prompt: %d, Completion: %d, Total: %d", 
                      content$usage$prompt_tokens, 
                      content$usage$completion_tokens, 
                      content$usage$total_tokens))
    }
    
    write_log(sprintf("Got response with %d lines", length(res)))
    write_log(sprintf("Raw response from Grok:\n%s", paste(res, collapse = "\n")))
    
    res
  }, simplify = FALSE)
  
  write_log("All chunks processed successfully")
  return(gsub(',$', '', unlist(allres)))
}
