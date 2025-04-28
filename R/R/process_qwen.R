#' Process request using QWEN models
#' @keywords internal
process_qwen <- function(prompt, model, api_key) {
  write_log(sprintf("Starting QWEN API request with model: %s", model))
  
  # Determine the appropriate API endpoint based on the key
  # Try international endpoint first, if it fails, try mainland endpoint
  international_endpoint <- "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
  mainland_endpoint <- "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  
  # Function to detect API key type
  detect_api_endpoint <- function(api_key) {
    # First try with international endpoint
    test_url <- international_endpoint
    test_body <- list(
      model = model,
      max_tokens = 10,
      messages = list(list(role = "user", content = "test"))
    )
    
    test_response <- tryCatch({
      httr::POST(
        url = test_url,
        httr::add_headers(
          "Authorization" = paste("Bearer", api_key),
          "Content-Type" = "application/json"
        ),
        body = jsonlite::toJSON(test_body, auto_unbox = TRUE),
        encode = "json"
      )
    }, error = function(e) {
      write_log(sprintf("Error testing international endpoint: %s", e$message))
      return(NULL)
    })
    
    # Check if international endpoint works
    if (!is.null(test_response) && !httr::http_error(test_response)) {
      write_log("International API endpoint detected and working")
      return(international_endpoint)
    }
    
    # If international fails, try mainland endpoint
    write_log("International endpoint failed, trying mainland endpoint")
    test_url <- mainland_endpoint
    
    test_response <- tryCatch({
      httr::POST(
        url = test_url,
        httr::add_headers(
          "Authorization" = paste("Bearer", api_key),
          "Content-Type" = "application/json"
        ),
        body = jsonlite::toJSON(test_body, auto_unbox = TRUE),
        encode = "json"
      )
    }, error = function(e) {
      write_log(sprintf("Error testing mainland endpoint: %s", e$message))
      return(NULL)
    })
    
    # Check if mainland endpoint works
    if (!is.null(test_response) && !httr::http_error(test_response)) {
      write_log("Mainland API endpoint detected and working")
      return(mainland_endpoint)
    }
    
    # If both fail, default to international endpoint and let the main function handle the error
    write_log("Both endpoints failed, defaulting to international endpoint")
    return(international_endpoint)
  }
  
  # Detect and use the appropriate endpoint
  url <- detect_api_endpoint(api_key)
  write_log(sprintf("Using endpoint: %s", url))
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