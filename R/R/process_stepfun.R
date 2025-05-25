#' Process request using Stepfun models
#' @keywords internal
process_stepfun <- function(prompt, model, api_key) {
  write_log("\n=== Starting Stepfun API Request ===\n")
  write_log(sprintf("Model: %s", model))

  # Stepfun API endpoint
  url <- "https://api.stepfun.com/v1/chat/completions"
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
      model = model,
      messages = list(
        list(
          role = "user",
          content = chunk_content
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
        "Content-Type" = "application/json",
        "Authorization" = paste("Bearer", api_key)
      ),
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      encode = "json"
    )

    # Check for errors
    if (httr::http_error(response)) {
      error_message <- httr::content(response, "parsed")
      write_log(sprintf("ERROR: Stepfun API request failed: %s",
                       if (!is.null(error_message$error$message)) error_message$error$message else "Unknown error"))
      return(NULL)
    }

    write_log("Parsing API response...")
    # Parse the response
    content <- httr::content(response, "parsed")

    # Check if response has the expected structure
    if (is.null(content) || is.null(content$choices) || length(content$choices) == 0 ||
        is.null(content$choices[[1]]$message) || is.null(content$choices[[1]]$message$content)) {
      write_log("ERROR: Unexpected response format from Stepfun API")
      write_log(sprintf("Content structure: %s", paste(names(content), collapse = ", ")))
      if (!is.null(content$choices)) {
        write_log(sprintf("Choices structure: %s", jsonlite::toJSON(content$choices, auto_unbox = TRUE, pretty = TRUE)))
      }
      return(NULL)
    }

    # Stepfun's response should be in content$choices[[1]]$message$content
    response_content <- content$choices[[1]]$message$content
    if (!is.character(response_content)) {
      write_log("ERROR: Response content is not a character string")
      write_log(sprintf("Response content type: %s", typeof(response_content)))
      write_log(sprintf("Response content structure: %s", jsonlite::toJSON(content$choices[[1]]$message, auto_unbox = TRUE, pretty = TRUE)))
      return(c("Error: Invalid response format"))
    }

    # Stepfun's response is in content$choices[[1]]$message$content
    tryCatch({
      res <- strsplit(response_content, '\n')[[1]]
      write_log(sprintf("Got response with %d lines", length(res)))
      write_log(sprintf("Raw response from Stepfun:\n%s", paste(res, collapse = "\n")))
      res
    }, error = function(e) {
      write_log(sprintf("ERROR: Failed to split response content: %s", e$message))
      return(c("Error: Failed to parse response"))
    })
  }, simplify = FALSE)

  write_log("All chunks processed successfully")

  # Filter out NULL values and handle errors more gracefully
  valid_results <- allres[!sapply(allres, is.null)]
  if (length(valid_results) == 0) {
    write_log("ERROR: No valid responses received from Stepfun")
    return(c("Error: No valid responses"))
  }

  return(gsub(',$', '', unlist(valid_results)))
}
