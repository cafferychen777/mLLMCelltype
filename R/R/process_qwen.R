#' Process request using QWEN models
#' @keywords internal
process_qwen <- function(prompt, model, api_key) {
  write_log(sprintf("Starting QWEN API request with model: %s", model))
  
  # QWEN API endpoint
  url <- "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
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
        "Authorization" = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      ),
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      encode = "json"
    )
    
    # Check for errors
    if (httr::http_error(response)) {
      error_message <- httr::content(response, "parsed")
      write_log(sprintf("ERROR: QWEN API request failed: %s", error_message$error$message))
      stop("QWEN API request failed: ", error_message$error$message)
    }
    
    write_log("Parsing API response...")
    # Parse the response
    content <- httr::content(response, "parsed")
    res <- strsplit(content$choices[[1]]$message$content, '\n')[[1]]
    write_log(sprintf("Got response with %d lines", length(res)))
    write_log(sprintf("Raw response from QWEN:\n%s", paste(res, collapse = "\n")))
    
    res
  }, simplify = FALSE)
  
  write_log("All chunks processed successfully")
  return(gsub(',$', '', unlist(allres)))
}