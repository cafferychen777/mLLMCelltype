#' Process request using DeepSeek models
#' @keywords internal
process_deepseek <- function(prompt, model, api_key) {
  write_log(sprintf("Starting DeepSeek API request with model: %s", model))

  # DeepSeek API endpoint
  url <- "https://api.deepseek.com/v1/chat/completions"
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
      messages = list(
        list(
          role = "system",
          content = "You are a cell type annotation expert. Based on the gene markers provided, identify the most likely cell type. Only provide the cell type name, without any additional explanation or numbering."
        ),
        list(
          role = "user",
          content = paste(input_lines[id], collapse = '\n')
        )
      ),
      stream = FALSE
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
      write_log(sprintf("ERROR: DeepSeek API request failed: %s", error_message$error$message))
      stop("DeepSeek API request failed: ", error_message$error$message)
    }

    write_log("Parsing API response...")
    # Parse the response
    content <- httr::content(response, "parsed")

    # Check if response has the expected structure
    if (is.null(content) || is.null(content$choices) || length(content$choices) == 0 ||
        is.null(content$choices[[1]]$message) || is.null(content$choices[[1]]$message$content)) {
      write_log("ERROR: Unexpected response format from DeepSeek API")
      write_log(sprintf("Content structure: %s", paste(names(content), collapse = ", ")))
      if (!is.null(content$choices)) {
        write_log(sprintf("Choices structure: %s", jsonlite::toJSON(content$choices, auto_unbox = TRUE, pretty = TRUE)))
      }
      return(NULL)
    }

    # DeepSeek's response should be in content$choices[[1]]$message$content
    response_content <- content$choices[[1]]$message$content
    if (!is.character(response_content)) {
      write_log("ERROR: Response content is not a character string")
      write_log(sprintf("Response content type: %s", typeof(response_content)))
      write_log(sprintf("Response content structure: %s", jsonlite::toJSON(content$choices[[1]]$message, auto_unbox = TRUE, pretty = TRUE)))
      return(c("Error: Invalid response format"))
    }

    # DeepSeek's response is in content$choices[[1]]$message$content
    tryCatch({
      res <- strsplit(response_content, '\n')[[1]]
      write_log(sprintf("Got response with %d lines", length(res)))
      write_log(sprintf("Raw response from DeepSeek:\n%s", paste(res, collapse = "\n")))
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
    write_log("ERROR: No valid responses received from DeepSeek")
    return(c("Error: No valid responses"))
  }

  return(gsub(',$', '', unlist(valid_results)))
}