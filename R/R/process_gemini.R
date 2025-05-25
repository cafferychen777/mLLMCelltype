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

    # Add robust error handling for response structure
    tryCatch({
      # Check if the response has the expected structure
      if (is.null(content$candidates) || length(content$candidates) == 0) {
        write_log("ERROR: Unexpected response structure - no candidates")
        write_log(sprintf("Response content: %s", jsonlite::toJSON(content, auto_unbox = TRUE, pretty = TRUE)))
        return("Error: Unexpected API response structure")
      }

      candidate <- content$candidates[[1]]

      # For Gemini 1.0 models
      if (!is.null(candidate$content$parts[[1]]$text)) {
        text_content <- candidate$content$parts[[1]]$text
        if (is.character(text_content)) {
          res <- strsplit(text_content, '\n')[[1]]
        } else {
          write_log("ERROR: Text content is not a character string")
          return("Error: Invalid text content format")
        }
      }
      # For Gemini 1.5/2.5 models (may have different structure)
      else if (!is.null(candidate$text)) {
        text_content <- candidate$text
        if (is.character(text_content)) {
          res <- strsplit(text_content, '\n')[[1]]
        } else {
          write_log("ERROR: Text content is not a character string")
          return("Error: Invalid text content format")
        }
      }
      # Try other possible response structures
      else if (!is.null(candidate$content$text)) {
        text_content <- candidate$content$text
        if (is.character(text_content)) {
          res <- strsplit(text_content, '\n')[[1]]
        } else {
          write_log("ERROR: Text content is not a character string")
          return("Error: Invalid text content format")
        }
      }
      else if (!is.null(content$text)) {
        text_content <- content$text
        if (is.character(text_content)) {
          res <- strsplit(text_content, '\n')[[1]]
        } else {
          write_log("ERROR: Text content is not a character string")
          return("Error: Invalid text content format")
        }
      }
      else {
        # If we can't find the text in any expected location, log the structure and return an error
        write_log("ERROR: Could not find text in response")
        write_log(sprintf("Response structure: %s", jsonlite::toJSON(content, auto_unbox = TRUE, pretty = TRUE)))
        return("Error: Could not parse API response")
      }
    }, error = function(e) {
      write_log(sprintf("ERROR in parsing Gemini response: %s", e$message))
      write_log(sprintf("Response content: %s", jsonlite::toJSON(content, auto_unbox = TRUE, pretty = TRUE)))
      return("Error: Failed to parse API response")
    })
    write_log(sprintf("Got response with %d lines", length(res)))
    write_log(sprintf("Raw response from Gemini:\n%s", paste(res, collapse = "\n")))

    res
  }, simplify = FALSE)

  write_log("All chunks processed successfully")

  # Filter out NULL values and handle errors more gracefully
  valid_results <- allres[!sapply(allres, is.null)]
  if (length(valid_results) == 0) {
    write_log("ERROR: No valid responses received from Gemini")
    return(c("Error: No valid responses"))
  }

  return(gsub(',$', '', unlist(valid_results)))
}