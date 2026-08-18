# Internal failure markers that must never become user-facing annotations.
.SENTINEL_VALUES <- c(
  "Parsing_Failed",
  "Insufficient_Responses",
  "Prediction_Missing",
  "Annotation_Missing",
  "Unknown"
)

.UNKNOWN_ANNOTATION_TOKENS <- c(
  tolower(.SENTINEL_VALUES),
  "unk",
  "n/a",
  "na",
  "none",
  "null",
  "nan",
  "-",
  "--",
  "inconclusive"
)

.unwrap_annotation_wrappers <- function(annotation) {
  wrapper_pairs <- c(
    "[" = "]",
    "(" = ")",
    "{" = "}",
    "\"" = "\"",
    "'" = "'"
  )
  normalized <- trimws(annotation)

  while (nchar(normalized) >= 2) {
    opening <- substr(normalized, 1, 1)
    expected_closing <- unname(wrapper_pairs[opening])
    if (is.na(expected_closing) ||
        substr(normalized, nchar(normalized), nchar(normalized)) != expected_closing) {
      break
    }
    normalized <- trimws(substr(normalized, 2, nchar(normalized) - 1))
  }
  normalized
}

is_real_cell_type_annotation <- function(annotation) {
  is_text_scalar <- is.character(annotation) &&
    length(annotation) == 1 &&
    !is.na(annotation)
  if (!is_text_scalar) {
    return(FALSE)
  }

  normalized <- tolower(.unwrap_annotation_wrappers(annotation))
  if (!nzchar(normalized) || normalized %in% .UNKNOWN_ANNOTATION_TOKENS) {
    return(FALSE)
  }

  contextual_unknown <- grepl(
    "^(unknown|inconclusive)\\s*[\\(\\[\\{].*[\\)\\]\\}]$",
    normalized,
    perl = TRUE
  )
  error_sentinel <- grepl(
    "^error(\\s*:.*|\\s*\\(.*\\))?$",
    normalized,
    perl = TRUE
  )
  !contextual_unknown && !error_sentinel
}

is_error_response <- function(result) {
  if (!is.character(result) || length(result) == 0) {
    return(FALSE)
  }
  non_empty <- result[!is.na(result) & nzchar(trimws(result))]
  if (length(non_empty) == 0) {
    return(FALSE)
  }
  # A response is an error only when it carries an error marker AND offers no
  # usable annotation. This preserves a valid multi-cluster response that flags
  # a single uncertain cluster (e.g. "Error: markers ambiguous") instead of
  # discarding every annotation the model returned.
  has_error <- any(grepl("^\\s*error\\s*:", non_empty, ignore.case = TRUE))
  has_error && !any(vapply(non_empty, is_real_cell_type_annotation, logical(1)))
}

normalize_model_response_lines <- function(result) {
  if (!is.character(result) || length(result) == 0 || anyNA(result)) {
    stop("Model response must be non-missing text")
  }

  lines <- unlist(strsplit(result, "\n", fixed = TRUE), use.names = FALSE)
  lines <- trimws(lines)
  lines <- sub(",+$", "", lines)
  lines <- trimws(lines)
  lines <- unname(lines[nzchar(lines)])

  if (length(lines) == 0) {
    stop("Model response must contain non-empty text")
  }
  if (is_error_response(lines)) {
    stop("Model response contains an error sentinel")
  }
  lines
}

normalize_token_usage <- function(usage) {
  if (!is.list(usage)) {
    return(NULL)
  }

  normalize_count <- function(value) {
    valid <- is.numeric(value) && length(value) == 1 &&
      !is.na(value) && is.finite(value) && value >= 0 &&
      value == floor(value)
    if (valid) as.numeric(value) else NULL
  }

  normalized <- list(
    prompt_tokens = normalize_count(usage$prompt_tokens),
    completion_tokens = normalize_count(usage$completion_tokens),
    total_tokens = normalize_count(usage$total_tokens)
  )

  cost <- usage$cost
  if (is.numeric(cost) && length(cost) == 1 &&
      !is.na(cost) && is.finite(cost) && cost >= 0) {
    normalized$cost <- as.numeric(cost)
  }
  normalized
}

is_valid_model_response <- function(result) {
  isTRUE(tryCatch({
    normalize_model_response_lines(result)
    TRUE
  }, error = function(e) FALSE))
}

is_retryable_http_status <- function(status_code) {
  is.numeric(status_code) &&
    length(status_code) == 1 &&
    !is.na(status_code) &&
    is.finite(status_code) &&
    status_code == floor(status_code) &&
    (status_code %in% c(408, 425, 429) || status_code >= 500)
}

get_api_request_timeout <- function() {
  timeout_seconds <- getOption("mLLMCelltype.api_timeout", 120)
  valid_timeout <- is.numeric(timeout_seconds) &&
    length(timeout_seconds) == 1 &&
    !is.na(timeout_seconds) &&
    is.finite(timeout_seconds) &&
    timeout_seconds > 0

  if (!valid_timeout) {
    stop("Option 'mLLMCelltype.api_timeout' must be one positive number of seconds")
  }

  as.numeric(timeout_seconds)
}

#' Extract the first JSON document from a model response
#'
#' Strips surrounding markdown fences and returns the first parseable JSON
#' array or object found in the text.
#'
#' @param text Single character string containing the model response.
#' @return The extracted JSON text as a character string.
#' @noRd
extract_json_from_response <- function(text) {
  if (!is.character(text) || length(text) != 1 || is.na(text)) {
    stop("Response text must be a single non-missing character string")
  }

  text <- paste(text, collapse = "\n")
  text <- trimws(text)

  # Strip markdown code fences if present
  text <- sub("^```[[:alnum:]]*\\s*\n?", "", text)
  text <- sub("\n?```\\s*$", "", text)
  text <- trimws(text)

  # Try parsing the whole text first
  parsed <- tryCatch(
    jsonlite::fromJSON(text, simplifyVector = FALSE),
    error = function(e) NULL
  )
  if (!is.null(parsed)) {
    return(text)
  }

  # Extract between first JSON delimiter and a matching closing delimiter
  first_bracket <- regexpr("\\[", text)
  first_brace <- regexpr("\\{", text)

  if (first_bracket > 0 && (first_brace < 0 || first_bracket < first_brace)) {
    start <- first_bracket
    end_positions <- gregexpr("\\]", text)[[1]]
  } else if (first_brace > 0) {
    start <- first_brace
    end_positions <- gregexpr("\\}", text)[[1]]
  } else {
    stop("No JSON array or object found in response")
  }

  end_positions <- end_positions[end_positions > start]
  if (length(end_positions) == 0) {
    stop("No complete JSON array or object found in response")
  }

  for (end in sort(end_positions, decreasing = TRUE)) {
    candidate <- substring(text, start, end)
    parsed <- tryCatch(
      jsonlite::fromJSON(candidate, simplifyVector = FALSE),
      error = function(e) NULL
    )
    if (!is.null(parsed)) {
      return(candidate)
    }
  }

  stop("Could not parse JSON array or object from response")
}

#' Parse reasoning annotation JSON into a structured named list
#'
#' @param json_text JSON text containing reasoning annotations.
#' @param expected_clusters Character vector of expected cluster IDs.
#' @return Named list mapping each expected cluster ID to a list with
#'   `cell_type`, `marker_genes`, and `gene_expression`.
#' @noRd
parse_reasoning_annotations <- function(json_text, expected_clusters) {
  json_text <- extract_json_from_response(json_text)

  parsed <- tryCatch(
    jsonlite::fromJSON(json_text, simplifyVector = FALSE),
    error = function(e) stop(paste("Failed to parse reasoning JSON response:", e$message))
  )

  if (!is.list(parsed)) {
    stop("Reasoning response must be a JSON array or object")
  }

  # Normalize array or object form into a list of records
  if (is.null(names(parsed))) {
    records <- parsed
  } else {
    records <- lapply(names(parsed), function(cid) {
      record <- parsed[[cid]]
      if (is.list(record)) {
        record$cluster_id <- cid
      }
      record
    })
  }

  result <- list()
  for (cid in expected_clusters) {
    result[[cid]] <- list(
      cell_type = "Unknown",
      marker_genes = "",
      gene_expression = ""
    )
  }

  for (record in records) {
    if (!is.list(record) || is.null(record$cluster_id)) {
      next
    }
    cid <- trimws(as.character(record$cluster_id))
    if (!(cid %in% expected_clusters)) {
      next
    }

    get_field <- function(field) {
      value <- record[[field]]
      if (!is.null(value) && is.character(value) && length(value) == 1 && !is.na(value)) {
        trimws(value)
      } else {
        ""
      }
    }

    result[[cid]] <- list(
      cell_type = get_field("cell_type"),
      marker_genes = get_field("marker_genes"),
      gene_expression = get_field("gene_expression")
    )
  }

  result
}

stop_api_request_error <- function(message, status_code = NULL,
                                   retryable = NULL) {
  if (is.null(retryable)) {
    retryable <- is_retryable_http_status(status_code)
  }
  condition <- structure(
    list(
      message = message,
      call = NULL,
      status_code = status_code,
      retryable = isTRUE(retryable)
    ),
    class = c("mllm_api_error", "error", "condition")
  )
  stop(condition)
}
