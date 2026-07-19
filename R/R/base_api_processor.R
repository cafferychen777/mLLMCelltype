#' Base API Processor Class
#'
#' Abstract base class for API processors that provides common functionality
#' including unified logging, error handling, input processing, and response validation.
#' This eliminates code duplication across all provider-specific processors.
#'
#' @export
BaseAPIProcessor <- R6::R6Class("BaseAPIProcessor",
  public = list(
    #' @field provider_name Name of the API provider
    provider_name = NULL,

    #' @field logger Unified logger instance
    logger = NULL,

    #' @field base_url Custom base URL for API endpoints
    base_url = NULL,

    #' @description
    #' Initialize the base API processor
    #' @param provider_name Provider identifier used for logging and dispatch
    #' @param base_url Optional custom API endpoint
    initialize = function(provider_name, base_url = NULL) {
      self$provider_name <- .normalize_required_string(provider_name, "provider_name")
      self$base_url <- resolve_provider_base_url(self$provider_name, base_url)
      self$logger <- get_logger()
      self$logger$info(sprintf("Initialized %s processor", self$provider_name),
                      list(provider = self$provider_name,
                           custom_url = !is.null(self$base_url)))
    },
    
    #' @description
    #' Main entry point for processing API requests
    #' @param prompt Prompt text to send
    #' @param model Model identifier
    #' @param api_key Provider API key
    process_request = function(prompt, model, api_key) {
      start_time <- Sys.time()

      tryCatch({
        inputs <- private$validate_inputs(prompt, model, api_key)
        prompt <- inputs$prompt
        model <- inputs$model
        api_key <- inputs$api_key

        self$logger$info(sprintf("Starting %s API request", self$provider_name),
                        list(model = model, provider = self$provider_name))

        # Make the API call and extract response
        call_result <- private$call_and_extract(prompt, model, api_key)
        final_result <- call_result$response
        
        # Log final status using semantic success (not just exception status)
        duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        semantic_success <- private$is_successful_result(final_result)
        self$logger$log_api_call(
          self$provider_name,
          model,
          duration,
          semantic_success,
          tokens = call_result$usage
        )
        
        return(final_result)
        
      }, error = function(e) {
        duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        self$logger$log_api_call(self$provider_name, model, duration, FALSE)
        self$logger$error(sprintf("%s API request failed: %s", self$provider_name, e$message),
                         list(provider = self$provider_name, model = model, error = e$message))
        if (inherits(e, "mllm_api_error")) {
          stop(e)
        }
        if (inherits(e, "curl_error")) {
          stop_api_request_error(
            sprintf("%s API request failed: %s", self$provider_name, e$message),
            retryable = TRUE
          )
        }
        stop(sprintf("%s API request failed: %s", self$provider_name, e$message))
      })
    },

    #' @description
    #' Get the API URL to use for requests
    #
    get_api_url = function() {
      if (!is.null(self$base_url)) {
        self$logger$debug("Using custom base URL",
                         list(provider = self$provider_name, url = self$base_url))
        return(self$base_url)
      }
      return(self$get_default_api_url())
    },

    #' @description
    #' Abstract method to be implemented by subclasses for getting default API URL
    #
    get_default_api_url = function() {
      stop("get_default_api_url must be implemented by subclass")
    },

    #' @description
    #' Abstract method to be implemented by subclasses for making the actual API call
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key Provider API key
    make_api_call = function(chunk_content, model, api_key) {
      stop("make_api_call must be implemented by subclass")
    },
    
    #' @description
    #' Abstract method to be implemented by subclasses for extracting content from response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      stop("extract_response_content must be implemented by subclass")
    },

    #' @description
    #' Extract normalized token usage from a provider response
    #' @param response HTTP response object
    extract_usage = function(response) {
      private$extract_usage_fields(response)
    }
  ),
  
  private = list(
    normalize_processor_input = function(value, argument_name, error_message) {
      tryCatch(
        .normalize_required_string(value, argument_name),
        error = function(e) {
          self$logger$error(
            sprintf("%s is missing, empty, or not a character scalar", argument_name),
            list(provider = self$provider_name, argument = argument_name)
          )
          stop(error_message)
        }
      )
    },

    # Validate and normalize input parameters
    validate_inputs = function(prompt, model, api_key) {
      normalized <- list(
        api_key = private$normalize_processor_input(
          api_key,
          "API key",
          sprintf("%s API key is required but not provided", self$provider_name)
        ),
        prompt = private$normalize_processor_input(
          prompt,
          "Prompt",
          "Prompt is required but not provided"
        ),
        model = private$normalize_processor_input(
          model,
          "Model",
          "Model is required but not provided"
        )
      )
      
      self$logger$debug("Input validation passed",
                       list(provider = self$provider_name, model = normalized$model))
      normalized
    },
    
    #' Make API call and extract response content
    #
    #
    #
    call_and_extract = function(prompt, model, api_key) {
      # Track progress through stages so the error handler knows what failed
      response <- NULL

      tryCatch({
        # Stage 1: API call
        response <- self$make_api_call(prompt, model, api_key)
        # Stage 2: Response extraction
        content <- self$extract_response_content(response, model)
      }, error = function(e) {
        # Unified audit log for failures at any stage
        self$logger$log_api_request_response(
          provider = self$provider_name,
          model = model,
          prompt_content = prompt,
          response_content = paste0("ERROR: ", e$message),
          request_metadata = list(provider = self$provider_name, failed = TRUE),
          response_metadata = list(
            error = e$message,
            stage = if (is.null(response)) "api_call" else "response_extraction"
          )
        )
        stop(e)
      })

      normalized_content <- tryCatch(
        normalize_model_response_lines(content),
        error = function(e) NULL
      )
      if (is.null(normalized_content)) {
        response_type <- typeof(content)
        self$logger$log_api_request_response(
          provider = self$provider_name,
          model = model,
          prompt_content = prompt,
          response_content = paste0("ERROR: Invalid model response (", response_type, ")"),
          request_metadata = list(provider = self$provider_name, failed = TRUE),
          response_metadata = list(
            error = "Invalid response format",
            stage = "response_validation",
            response_type = response_type
          )
        )
        stop("Invalid response format from API")
      }

      # Log successful request and response for audit/debugging
      self$logger$log_api_request_response(
        provider = self$provider_name,
        model = model,
        prompt_content = prompt,
        response_content = content,
        request_metadata = list(provider = self$provider_name),
        response_metadata = list(
          raw_response_class = class(response),
          extracted_content_length = sum(nchar(content))
        )
      )

      self$logger$debug(sprintf("Processed response from %s", self$provider_name),
                       list(provider = self$provider_name,
                            model = model,
                            lines_count = length(normalized_content),
                            response_length = sum(nchar(content))))
      usage <- tryCatch(
        self$extract_usage(response),
        error = function(e) {
          if (is.function(self$logger$warn)) {
            self$logger$warn("Failed to extract token usage", list(
              provider = self$provider_name,
              error = e$message
            ))
          }
          NULL
        }
      )
      return(list(response = normalized_content, usage = usage))
    },

    build_chat_completions_body = function(chunk_content, model, extra = list()) {
      body <- list(
        model = model,
        messages = list(
          list(
            role = "user",
            content = chunk_content
          )
        )
      )

      if (length(extra) > 0) {
        for (field in names(extra)) {
          body[[field]] <- extra[[field]]
        }
      }

      body
    },

    extract_error_message = function(response) {
      error_content <- tryCatch(
        httr::content(response, "parsed"),
        error = function(e) NULL
      )
      is_non_empty_string <- function(value) {
        is.character(value) && length(value) == 1 && !is.na(value) && nzchar(value)
      }

      if (is.list(error_content) &&
          is.list(error_content$error) &&
          is_non_empty_string(error_content$error$message)) {
        return(error_content$error$message)
      }
      if (is.list(error_content) &&
          is_non_empty_string(error_content$error)) {
        return(error_content$error)
      }
      if (is.list(error_content) &&
          is_non_empty_string(error_content$message)) {
        return(error_content$message)
      }

      sprintf("HTTP %d error", httr::status_code(response))
    },

    stop_for_http_error = function(
      response,
      model,
      provider_label = get_builtin_provider_display_name(self$provider_name)
    ) {
      status_code <- httr::status_code(response)
      if (identical(as.integer(status_code), 200L)) {
        return(invisible(NULL))
      }

      error_message <- private$extract_error_message(response)

      self$logger$error(sprintf("%s API request failed", provider_label),
                       list(error = error_message,
                            provider = self$provider_name,
                            model = model,
                            status_code = status_code))

      stop_api_request_error(
        sprintf("%s API request failed: %s", provider_label, error_message),
        status_code = status_code
      )
    },

    post_chat_completions_request = function(chunk_content,
                                            model,
                                            api_key,
                                            api_url = self$get_api_url(),
                                            body_extra = list(),
                                            headers = list(),
                                            provider_label = get_builtin_provider_display_name(
                                              self$provider_name
                                            )) {
      body <- private$build_chat_completions_body(
        chunk_content = chunk_content,
        model = model,
        extra = body_extra
      )

      self$logger$debug(sprintf("Sending API request to %s", provider_label),
                       list(model = model, provider = self$provider_name))

      request_headers <- c(
        list(
          "Authorization" = paste("Bearer", api_key),
          "Content-Type" = "application/json"
        ),
        headers
      )

      response <- httr::POST(
        url = api_url,
        do.call(httr::add_headers, request_headers),
        body = body,
        encode = "json",
        httr::timeout(30)
      )

      private$stop_for_http_error(response, model, provider_label)
      response
    },

    extract_chat_completions_content = function(response,
                                               model,
                                               provider_label = get_builtin_provider_display_name(
                                                 self$provider_name
                                               )) {
      self$logger$debug(sprintf("Parsing %s API response", provider_label),
                       list(provider = self$provider_name, model = model))

      content <- httr::content(response, "parsed")

      if (is.null(content) ||
          is.null(content$choices) ||
          length(content$choices) == 0 ||
          is.null(content$choices[[1]]$message) ||
          is.null(content$choices[[1]]$message$content)) {

        self$logger$error(sprintf("Unexpected response format from %s API", provider_label),
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              choices_available = !is.null(content$choices),
                              choices_count = if (!is.null(content$choices)) {
                                length(content$choices)
                              } else {
                                0
                              }))

        stop(sprintf("Unexpected response format from %s API", provider_label))
      }

      if (identical(content$choices[[1]]$finish_reason, "length")) {
        self$logger$warn(
          sprintf(
            "%s response was truncated (finish_reason='length'); trailing clusters may be marked Unknown",
            provider_label
          ),
          list(provider = self$provider_name, model = model)
        )
      }

      content$choices[[1]]$message$content
    },

    extract_usage_fields = function(response,
                                    usage_field = "usage",
                                    prompt_field = "prompt_tokens",
                                    completion_field = "completion_tokens",
                                    total_field = "total_tokens",
                                    cost_field = "cost",
                                    derive_total = FALSE) {
      if (!inherits(response, "response")) {
        return(NULL)
      }
      content <- tryCatch(
        httr::content(response, "parsed"),
        error = function(e) NULL
      )
      if (!is.list(content)) {
        return(NULL)
      }
      usage <- content[[usage_field, exact = TRUE]]
      if (!is.list(usage)) {
        return(NULL)
      }

      field_value <- function(field) {
        if (is.null(field)) NULL else usage[[field, exact = TRUE]]
      }
      normalized <- normalize_token_usage(list(
        prompt_tokens = field_value(prompt_field),
        completion_tokens = field_value(completion_field),
        total_tokens = field_value(total_field),
        cost = field_value(cost_field)
      ))
      if (isTRUE(derive_total) &&
          !is.null(normalized$prompt_tokens) &&
          !is.null(normalized$completion_tokens)) {
        normalized$total_tokens <- normalized$prompt_tokens +
          normalized$completion_tokens
      }
      normalized
    },

    is_successful_result = function(result) {
      is_valid_model_response(result)
    }
  )
)
