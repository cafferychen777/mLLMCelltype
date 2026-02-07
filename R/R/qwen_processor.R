#' Qwen API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Qwen models.
#' Handles Qwen-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
# Package-level cache for Qwen endpoint (persists across QwenProcessor instances)
.qwen_endpoint_cache <- new.env(parent = emptyenv())

QwenProcessor <- R6::R6Class("QwenProcessor",
  inherit = BaseAPIProcessor,

  private = list(

    #' @description
    #' Test if an endpoint is accessible
    #
    #
    #
    test_endpoint = function(url, api_key) {
      tryCatch({
        # Simple test payload with correct Qwen format
        test_payload <- list(
          model = "qwen-turbo",
          input = list(
            messages = list(
              list(role = "user", content = "test")
            )
          ),
          parameters = list(
            max_tokens = 1,
            temperature = 0.1
          )
        )

        response <- httr::POST(
          url = url,
          httr::add_headers(
            "Authorization" = paste("Bearer", api_key),
            "Content-Type" = "application/json"
          ),
          body = jsonlite::toJSON(test_payload, auto_unbox = TRUE),
          encode = "json",
          httr::timeout(10)  # 10 second timeout for quick test
        )

        return(httr::status_code(response) == 200)
      }, error = function(e) {
        return(FALSE)
      })
    }
  ),

  public = list(
    #' @description
    #' Initialize Qwen processor
    #
    initialize = function(base_url = NULL) {
      super$initialize("qwen", base_url)
    },

    #' @description
    #' Get default Qwen API URL with intelligent endpoint selection
    #
    #' @details Qwen has two API endpoints:
    #'   - International: https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation (preferred)
    #'   - Domestic (China): https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation (fallback)
    #'   The processor automatically tries international first, then falls back to domestic if needed.
    get_default_api_url = function() {
      return("https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
    },

    #' @description
    #' Get working Qwen API URL with automatic endpoint detection
    #
    #
    get_working_api_url = function(api_key) {
      # Check package-level cache first (persists across instances)
      if (!is.null(.qwen_endpoint_cache$url)) {
        return(.qwen_endpoint_cache$url)
      }

      international_url <- "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
      domestic_url <- "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

      self$logger$debug("Testing Qwen endpoints for accessibility")

      # Try international endpoint first
      if (private$test_endpoint(international_url, api_key)) {
        self$logger$info("Using Qwen international endpoint", list(url = international_url))
        .qwen_endpoint_cache$url <- international_url
        return(international_url)
      }

      # Fallback to domestic endpoint
      if (private$test_endpoint(domestic_url, api_key)) {
        self$logger$info("Using Qwen domestic endpoint (international failed)", list(url = domestic_url))
        .qwen_endpoint_cache$url <- domestic_url
        return(domestic_url)
      }

      # If both fail, return international as default and let the main call handle the error
      self$logger$warn("Both Qwen endpoints failed during testing, using international as default")
      return(international_url)
    },
    
    #' @description
    #' Make API call to Qwen
    #
    #
    #
    #
    make_api_call = function(chunk_content, model, api_key) {
      # Prepare request body with proper Qwen format
      body <- list(
        model = model,
        input = list(
          messages = list(
            list(
              role = "user",
              content = chunk_content
            )
          )
        ),
        parameters = list(
          max_tokens = 2000,
          temperature = 0.1
        )
      )
      
      self$logger$debug("Sending API request to Qwen",
                       list(model = model, provider = self$provider_name))
      
      # Get API URL: custom base_url takes priority, otherwise auto-detect endpoint
      api_url <- if (!is.null(self$base_url)) {
        self$base_url
      } else {
        self$get_working_api_url(api_key)
      }

      # Make the API request
      response <- httr::POST(
        url = api_url,
        httr::add_headers(
          "Authorization" = paste("Bearer", api_key),
          "Content-Type" = "application/json"
        ),
        body = jsonlite::toJSON(body, auto_unbox = TRUE),
        encode = "json"
      )
      
      # Check for HTTP errors
      if (httr::http_error(response)) {
        error_content <- tryCatch(
          httr::content(response, "parsed"),
          error = function(e) NULL
        )
        error_message <- if (is.list(error_content) && !is.null(error_content$error$message)) {
          error_content$error$message
        } else {
          sprintf("HTTP %d error", httr::status_code(response))
        }
        
        self$logger$error("Qwen API request failed",
                         list(error = error_message,
                              provider = self$provider_name,
                              model = model,
                              status_code = httr::status_code(response)))
        
        stop(sprintf("Qwen API request failed: %s", error_message))
      }
      
      return(response)
    },
    
    #' @description
    #' Extract response content from Qwen API response
    #
    #
    #
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing Qwen API response",
                       list(provider = self$provider_name, model = model))

      # Parse the response
      content <- httr::content(response, "parsed")

      # Extract from Qwen's format: older models use output$text,
      # newer models (qwen3-*) use output$choices[[1]]$message$content
      if (!is.null(content$output$text)) {
        response_content <- content$output$text
      } else if (!is.null(content$output$choices) &&
                 length(content$output$choices) > 0 &&
                 !is.null(content$output$choices[[1]]$message$content)) {
        response_content <- content$output$choices[[1]]$message$content
      } else {
        self$logger$error("Unexpected response format from Qwen API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              output_keys = if (!is.null(content$output)) names(content$output) else NULL))
        stop("Unexpected response format from Qwen API")
      }

      return(response_content)
    }
  )
)