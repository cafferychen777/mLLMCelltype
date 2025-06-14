#' Process request using Anthropic models
#' @keywords internal
process_anthropic <- function(prompt, model, api_key) {
  log_info("Starting Anthropic API request", list(model = model, provider = "anthropic"))

  # Anthropic API endpoint
  url <- "https://api.anthropic.com/v1/messages"
  log_debug("Using model", list(model = model, provider = "anthropic"))

  # Process all input at once
  input_lines <- strsplit(prompt, "\n")[[1]]
  cutnum <- 1  # Changed to always use 1 chunk

  log_debug("Processing chunks of input", list(chunk_count = cutnum))

  if (cutnum > 1) {
    cid <- as.numeric(cut(1:length(input_lines), cutnum))
  } else {
    cid <- rep(1, length(input_lines))
  }

  # Process each chunk
  allres <- sapply(1:cutnum, function(i) {
    log_debug("Processing chunk", list(current_chunk = i, total_chunks = cutnum))
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

    log_debug("Sending API request", list(model = model, provider = "anthropic"))
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
      log_error("Invalid response object", list(provider = "anthropic", model = model))
      return(NULL)
    }

    if (response$status_code >= 400) {
      error_message <- httr::content(response, "parsed")
      log_error("Anthropic API request failed", list(
        provider = "anthropic",
        model = model,
        status_code = response$status_code,
        error = if (!is.null(error_message$error$message)) error_message$error$message else "Unknown error"
      ))
      return(NULL)
    }

    log_debug("Parsing API response", list(provider = "anthropic", model = model))
    # Parse the response
    content <- httr::content(response, "parsed")

    # Check if response has the expected structure
    if (is.null(content) || is.null(content$content) || length(content$content) == 0 ||
        is.null(content$content[[1]]$text)) {
      log_error("Unexpected response format from Anthropic API", list(
        provider = "anthropic",
        model = model,
        content_structure = names(content),
        content_available = !is.null(content$content),
        content_count = if(!is.null(content$content)) length(content$content) else 0
      ))
      return(NULL)
    }

    # Claude's response is in content$content[[1]]$text
    response_content <- content$content[[1]]$text
    if (!is.character(response_content)) {
      write_log("ERROR: Response content is not a character string")
      write_log(sprintf("Response content type: %s", typeof(response_content)))
      write_log(sprintf("Response content structure: %s", jsonlite::toJSON(content$content[[1]], auto_unbox = TRUE, pretty = TRUE)))
      return(c("Error: Invalid response format"))
    }

    tryCatch({
      res <- strsplit(response_content, '\n')[[1]]
      write_log(sprintf("Got response with %d lines", length(res)))
      write_log(sprintf("Raw response from Claude:\n%s", paste(res, collapse = "\n")))
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
    write_log("ERROR: No valid responses received from Anthropic")
    return(c("Error: No valid responses"))
  }

  return(gsub(',$', '', unlist(valid_results)))
}