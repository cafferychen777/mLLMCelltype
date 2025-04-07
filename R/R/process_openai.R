#' Process request using OpenAI models
#' @keywords internal
process_openai <- function(prompt, model, api_key) {
  write_log(sprintf("Starting OpenAI API request with model: %s", model))
  
  # Check if API key is provided and not empty
  if (is.null(api_key) || api_key == "") {
    write_log("ERROR: OpenAI API key is missing or empty")
    stop("OpenAI API key is required but not provided")
  }
  
  # OpenAI API endpoint
  url <- "https://api.openai.com/v1/chat/completions"
  write_log(sprintf("Using model: %s", model))
  
  # Process all input at once instead of chunks
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
      messages = list(
        list(
          role = "user",
          content = paste(input_lines[id], collapse = '\n')
        )
      ),
      store = TRUE
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
      write_log(sprintf("ERROR: OpenAI API request failed: %s", error_message$error$message))
      stop("OpenAI API request failed: ", error_message$error$message)
    }
    
    write_log("Parsing API response...")
    # Parse the response
    content <- httr::content(response, "parsed")
    res <- strsplit(content$choices[[1]]$message$content, '\n')[[1]]
    write_log(sprintf("Got response with %d lines", length(res)))
    write_log(sprintf("Raw response from OpenAI:\n%s", paste(res, collapse = "\n")))
    
    res
  }, simplify = FALSE)
  
  write_log("All chunks processed successfully")
  return(gsub(',$', '', unlist(allres)))
}