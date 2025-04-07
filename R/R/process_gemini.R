#' Process request using Gemini models
#' @keywords internal
process_gemini <- function(prompt, model, api_key) {
  write_log("\n=== Starting Gemini API Request ===\n")
  write_log(sprintf("Model: %s", model))
  
  # Gemini API endpoint
  base_url <- "https://generativelanguage.googleapis.com/v1beta/models"
  url <- sprintf("%s/%s:generateContent?key=%s", base_url, model, api_key)
  write_log("API URL:")
  write_log(url)
  
  # Process all input at once
  input_lines <- strsplit(prompt, "\n")[[1]]
  write_log("\nInput lines:")
  write_log(paste(input_lines, collapse = "\n"))
  
  cutnum <- 1  # Changed to always use 1 chunk
  write_log(sprintf("\nProcessing input in %d chunk(s)", cutnum))
  
  if (cutnum > 1) {
    cid <- as.numeric(cut(1:length(input_lines), cutnum))	
  } else {
    cid <- rep(1, length(input_lines))
  }
  
  # Process each chunk
  allres <- sapply(1:cutnum, function(i) {
    write_log(sprintf("\nProcessing chunk %d of %d", i, cutnum))
    id <- which(cid == i)
    
    chunk_content <- paste(input_lines[id], collapse = '\n')
    write_log("\nChunk content:")
    write_log(chunk_content)
    
    # Prepare the request body
    body <- list(
      contents = list(
        list(
          parts = list(
            list(
              text = chunk_content
            )
          )
        )
      )
    )
    
    write_log("\nRequest body:")
    write_log(jsonlite::toJSON(body, auto_unbox = TRUE, pretty = TRUE))
    
    write_log("\nSending API request...")
    # Make the API request
    response <- httr::POST(
      url = url,
      httr::add_headers(
        "Content-Type" = "application/json"
      ),
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      encode = "json"
    )
    
    # Check for errors
    if (httr::http_error(response)) {
      error_message <- httr::content(response, "parsed")
      write_log(sprintf("ERROR: Gemini API request failed: %s", 
                       if (!is.null(error_message$error$message)) error_message$error$message else "Unknown error"))
      return(NULL)
    }
    
    write_log("Parsing API response...")
    # Parse the response
    content <- httr::content(response, "parsed")
    # Gemini's response is in content$candidates[[1]]$content$parts[[1]]$text
    res <- strsplit(content$candidates[[1]]$content$parts[[1]]$text, '\n')[[1]]
    write_log(sprintf("Got response with %d lines", length(res)))
    write_log(sprintf("Raw response from Gemini:\n%s", paste(res, collapse = "\n")))
    
    res
  }, simplify = FALSE)
  
  write_log("All chunks processed successfully")
  return(gsub(',$', '', unlist(allres)))
}