#' Process request using OpenAI models
#' @keywords internal
process_openai <- function(prompt, model, api_key) {
  log_info("Starting OpenAI API request", list(model = model, provider = "openai"))

  # Check if API key is provided and not empty
  if (is.null(api_key) || api_key == "") {
    log_error("OpenAI API key is missing or empty", list(provider = "openai"))
    stop("OpenAI API key is required but not provided")
  }

  # OpenAI API endpoint
  url <- "https://api.openai.com/v1/chat/completions"
  log_debug("Using model", list(model = model, provider = "openai"))

  # Process all input at once instead of chunks
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
      messages = list(
        list(
          role = "user",
          content = paste(input_lines[id], collapse = '\n')
        )
      ),
      store = TRUE
    )

    log_debug("Sending API request", list(model = model, provider = "openai"))
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
      log_error("OpenAI API request failed", list(
        error = error_message$error$message,
        provider = "openai",
        model = model,
        status_code = httr::status_code(response)
      ))
      stop("OpenAI API request failed: ", error_message$error$message)
    }

    log_debug("Parsing API response", list(provider = "openai", model = model))
    # Parse the response
    content <- httr::content(response, "parsed")

    # Check if response has the expected structure
    if (is.null(content) || is.null(content$choices) || length(content$choices) == 0 ||
        is.null(content$choices[[1]]$message) || is.null(content$choices[[1]]$message$content)) {
      log_error("Unexpected response format from OpenAI API", list(
        provider = "openai",
        model = model,
        content_structure = names(content),
        choices_available = !is.null(content$choices),
        choices_count = if(!is.null(content$choices)) length(content$choices) else 0
      ))
      return(NULL)
    }

    # OpenAI's response should be in content$choices[[1]]$message$content
    response_content <- content$choices[[1]]$message$content
    if (!is.character(response_content)) {
      log_error("Response content is not a character string", list(
        provider = "openai",
        model = model,
        response_type = typeof(response_content),
        message_structure = names(content$choices[[1]]$message)
      ))
      return(c("Error: Invalid response format"))
    }

    tryCatch({
      res <- strsplit(response_content, '\n')[[1]]
      log_debug("Got response from OpenAI", list(
        provider = "openai",
        model = model,
        lines_count = length(res),
        response_length = nchar(response_content)
      ))
      res
    }, error = function(e) {
      log_error("Failed to split response content", list(
        provider = "openai",
        model = model,
        error = e$message
      ))
      return(c("Error: Failed to parse response"))
    })
  }, simplify = FALSE)

  log_info("All chunks processed successfully", list(
    provider = "openai",
    model = model,
    chunks_processed = cutnum
  ))

  # Filter out NULL values and handle errors more gracefully
  valid_results <- allres[!sapply(allres, is.null)]
  if (length(valid_results) == 0) {
    log_error("No valid responses received from OpenAI", list(
      provider = "openai",
      model = model,
      chunks_attempted = cutnum
    ))
    return(c("Error: No valid responses"))
  }

  return(gsub(',$', '', unlist(valid_results)))
}