#' Base API Processor Class
#' 
#' Abstract base class for API processors that provides common functionality
#' including unified logging, error handling, input processing, and response validation.
#' This eliminates code duplication across all provider-specific processors.
#'
#' @importFrom R6 R6Class
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
    #
    #
    initialize = function(provider_name, base_url = NULL) {
      self$provider_name <- provider_name
      self$base_url <- base_url
      self$logger <- get_logger()
      self$logger$info(sprintf("Initialized %s processor", provider_name),
                      list(provider = provider_name, custom_url = !is.null(base_url)))
    },
    
    #' @description
    #' Main entry point for processing API requests
    #
    #
    #
    #
    process_request = function(prompt, model, api_key) {
      start_time <- Sys.time()
      
      self$logger$info(sprintf("Starting %s API request", self$provider_name), 
                      list(model = model, provider = self$provider_name))
      
      tryCatch({
        # Validate inputs
        private$validate_inputs(prompt, model, api_key)

        # Make the API call and extract response
        final_result <- private$call_and_extract(prompt, model, api_key)
        
        # Log final status using semantic success (not just exception status)
        duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        semantic_success <- private$is_successful_result(final_result)
        self$logger$log_api_call(self$provider_name, model, duration, semantic_success)
        
        return(final_result)
        
      }, error = function(e) {
        duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        self$logger$log_api_call(self$provider_name, model, duration, FALSE)
        self$logger$error(sprintf("%s API request failed: %s", self$provider_name, e$message),
                         list(provider = self$provider_name, model = model, error = e$message))
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
    #
    #
    #
    #
    make_api_call = function(chunk_content, model, api_key) {
      stop("make_api_call must be implemented by subclass")
    },
    
    #' @description
    #' Abstract method to be implemented by subclasses for extracting content from response
    #
    #
    #
    extract_response_content = function(response, model) {
      stop("extract_response_content must be implemented by subclass")
    }
  ),
  
  private = list(
    # Validate input parameters
    validate_inputs = function(prompt, model, api_key) {
      api_key_missing <- is.null(api_key) ||
        length(api_key) != 1 ||
        is.na(api_key) ||
        !nzchar(trimws(as.character(api_key)))

      if (api_key_missing) {
        self$logger$error(sprintf("%s API key is missing or empty", self$provider_name),
                         list(provider = self$provider_name))
        stop(sprintf("%s API key is required but not provided", self$provider_name))
      }

      prompt_missing <- is.null(prompt) ||
        length(prompt) != 1 ||
        is.na(prompt) ||
        !nzchar(trimws(as.character(prompt)))

      if (prompt_missing) {
        self$logger$error("Prompt is missing or empty",
                         list(provider = self$provider_name))
        stop("Prompt is required but not provided")
      }

      model_missing <- is.null(model) ||
        length(model) != 1 ||
        is.na(model) ||
        !nzchar(trimws(as.character(model)))

      if (model_missing) {
        self$logger$error("Model is missing or empty",
                         list(provider = self$provider_name))
        stop("Model is required but not provided")
      }
      
      self$logger$debug("Input validation passed",
                       list(provider = self$provider_name, model = model))
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

      # Validate before logging success
      if (!is.character(content)) {
        self$logger$log_api_request_response(
          provider = self$provider_name,
          model = model,
          prompt_content = prompt,
          response_content = paste0("ERROR: Response is not character (", typeof(content), ")"),
          request_metadata = list(provider = self$provider_name, failed = TRUE),
          response_metadata = list(
            error = "Invalid response format",
            stage = "response_validation",
            response_type = typeof(content)
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

      res <- strsplit(content, "\n")[[1]]
      self$logger$debug(sprintf("Processed response from %s", self$provider_name),
                       list(provider = self$provider_name,
                            model = model,
                            lines_count = length(res),
                            response_length = nchar(content)))
      return(res)
    },

    is_successful_result = function(result) {
      if (is.null(result) || length(result) == 0) {
        return(FALSE)
      }
      if (!is.character(result)) {
        return(TRUE)
      }
      text <- paste(result, collapse = "\n")
      !grepl("^\\s*Error:", text, ignore.case = TRUE)
    }
  )
)
