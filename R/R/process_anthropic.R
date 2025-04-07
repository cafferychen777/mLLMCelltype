#' Process request using Anthropic models
#' @keywords internal
process_anthropic <- function(prompt, model, api_key) {
  write_log(sprintf("Starting Anthropic API request with model: %s", model))
  
  # Anthropic API endpoint
  url <- "https://api.anthropic.com/v1/messages"
  write_log(sprintf("Using model: %s", model))
  
  # Process all input at once
  input_lines <- strsplit(prompt, "\n")[[1]]
  cutnum <- 1  # Changed to always use 1 chunk
  
  write_log(sprintf("Processing %d chunks of input", cutnum))
  
  if (cutnum > 1) {
    cid <- as.numeric(cut(1:length(input_lines), cutnum))	
  } else {
    cid <- rep(1, length(input_lines))
  }
  
  # Process each chunk
  allres <- sapply(1:cutnum, function(i) {
    write_log(sprintf("Processing chunk %d of %d", i, cutnum))
    id <- which(cid == i)
    
    # Prepare the request body
    body <- list(
      model = model,
      max_tokens = 1024,
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
        "x-api-key" = api_key,
        "anthropic-version" = "2023-06-01",
        "content-type" = "application/json"
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
      write_log(sprintf("ERROR: Anthropic API request failed: %s", 
                       if (!is.null(error_message$error$message)) error_message$error$message else "Unknown error"))
      return(NULL)
    }
    
    write_log("Parsing API response...")
    # Parse the response
    content <- httr::content(response, "parsed")
    # Claude's response is in content$content[[1]]$text
    res <- strsplit(content$content[[1]]$text, '\n')[[1]]
    write_log(sprintf("Got response with %d lines", length(res)))
    write_log(sprintf("Raw response from Claude:\n%s", paste(res, collapse = "\n")))
    
    res
  }, simplify = FALSE)
  
  write_log("All chunks processed successfully")
  return(gsub(',$', '', unlist(allres)))
}