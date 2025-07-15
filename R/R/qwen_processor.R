#' Qwen API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Qwen models.
#' Handles Qwen-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
QwenProcessor <- R6::R6Class("QwenProcessor",
  inherit = BaseAPIProcessor,
  
  private = list(
    # Cache for the working endpoint to avoid repeated testing
    working_endpoint = NULL,

    #' @description
    #' Test if an endpoint is accessible
    #' @param url The endpoint URL to test
    #' @param api_key API key for authentication
    #' @return TRUE if accessible, FALSE otherwise
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
    #' @param base_url Optional custom base URL for Qwen API
    initialize = function(base_url = NULL) {
      super$initialize("qwen", base_url)
    },

    #' @description
    #' Get default Qwen API URL with intelligent endpoint selection
    #' @return Default Qwen API endpoint URL
    #' @details Qwen has two API endpoints:
    #'   - International: https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation (preferred)
    #'   - Domestic (China): https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation (fallback)
    #'   The processor automatically tries international first, then falls back to domestic if needed.
    get_default_api_url = function() {
      return("https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
    },

    #' @description
    #' Get working Qwen API URL with automatic endpoint detection
    #' @param api_key API key for testing endpoints
    #' @return Working Qwen API endpoint URL
    get_working_api_url = function(api_key) {
      # If we already found a working endpoint, use it
      if (!is.null(private$working_endpoint)) {
        return(private$working_endpoint)
      }

      international_url <- "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
      domestic_url <- "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

      self$logger$debug("Testing Qwen endpoints for accessibility")

      # Try international endpoint first
      if (private$test_endpoint(international_url, api_key)) {
        self$logger$info("Using Qwen international endpoint", list(url = international_url))
        private$working_endpoint <- international_url
        return(international_url)
      }

      # Fallback to domestic endpoint
      if (private$test_endpoint(domestic_url, api_key)) {
        self$logger$info("Using Qwen domestic endpoint (international failed)", list(url = domestic_url))
        private$working_endpoint <- domestic_url
        return(domestic_url)
      }

      # If both fail, return international as default and let the main call handle the error
      self$logger$warn("Both Qwen endpoints failed during testing, using international as default")
      return(international_url)
    },
    
    #' @description
    #' Make API call to Qwen
    #' @param chunk_content Content for this chunk
    #' @param model Model identifier
    #' @param api_key API key
    #' @return httr response object
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
      
      # Get working API URL (with intelligent endpoint selection)
      api_url <- if (!is.null(self$base_url)) {
        self$get_api_url()  # Use custom base_url if provided
      } else {
        self$get_working_api_url(api_key)  # Use intelligent endpoint selection
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
        error_content <- httr::content(response, "parsed")
        error_message <- if (!is.null(error_content$error$message)) {
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
    #' @param response httr response object
    #' @param model Model identifier
    #' @return Extracted text content
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing Qwen API response",
                       list(provider = self$provider_name, model = model))

      # Parse the response
      content <- httr::content(response, "parsed")

      # Check if response has the expected Qwen structure
      if (is.null(content) || is.null(content$output) || is.null(content$output$text)) {

        self$logger$error("Unexpected response format from Qwen API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              output_available = !is.null(content$output),
                              text_available = if(!is.null(content$output)) !is.null(content$output$text) else FALSE))

        stop("Unexpected response format from Qwen API")
      }

      # Extract the response content from Qwen's format
      response_content <- content$output$text

      return(response_content)
    }
  )
)