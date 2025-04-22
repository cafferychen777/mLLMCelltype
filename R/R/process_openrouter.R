#' Process request using OpenRouter API
#'
#' This function processes annotation requests using the OpenRouter API, which provides
#' access to various LLM models including OpenAI, Anthropic, Meta, and Google models.
#'
#' @param prompt Character string containing the prompt to send to the API
#' @param model Character string specifying the model to use (e.g., 'openai/gpt-4o')
#' @param api_key Character string containing the OpenRouter API key
#' @return Character vector containing the model's response
#' @keywords internal
#' @export
process_openrouter <- function(prompt, model, api_key) {
  write_log(sprintf("Starting OpenRouter API request with model: %s", model))
  
  # Check if API key is provided and not empty
  if (is.null(api_key) || api_key == "") {
    write_log("ERROR: OpenRouter API key is missing or empty")
    stop("OpenRouter API key is required but not provided")
  }
  
  # OpenRouter API endpoint
  url <- "https://openrouter.ai/api/v1/chat/completions"
  write_log(sprintf("Using model: %s", model))
  
  # Process all input at once
  input_lines <- strsplit(prompt, "\n")[[1]]
  cutnum <- 1  # Use 1 chunk for processing
  
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
        "Content-Type" = "application/json",
        "Authorization" = paste("Bearer", api_key)
      ),
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      encode = "json"
    )
    
    # Check for errors
    if (httr::http_error(response)) {
      error_message <- httr::content(response, "parsed")
      write_log(sprintf("ERROR: OpenRouter API request failed: %s", 
                       if (!is.null(error_message$error$message)) error_message$error$message else "Unknown error"))
      stop("OpenRouter API request failed: ", 
           if (!is.null(error_message$error$message)) error_message$error$message else "Unknown error")
    }
    
    write_log("Parsing API response...")
    # Parse the response
    content <- httr::content(response, "parsed")
    
    # OpenRouter follows OpenAI's response format
    res <- strsplit(content$choices[[1]]$message$content, '\n')[[1]]
    write_log(sprintf("Got response with %d lines", length(res)))
    write_log(sprintf("Raw response from OpenRouter:\n%s", paste(res, collapse = "\n")))
    
    res
  }, simplify = FALSE)
  
  write_log("All chunks processed successfully")
  return(gsub(',$', '', unlist(allres)))
}