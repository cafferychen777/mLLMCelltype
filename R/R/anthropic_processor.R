#' Anthropic API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Anthropic models.
#' Handles Anthropic-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
AnthropicProcessor <- R6::R6Class("AnthropicProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize Anthropic processor
    #
    initialize = function(base_url = NULL) {
      super$initialize("anthropic", base_url)
    },

    #' @description
    #' Get default Anthropic API URL
    #
    get_default_api_url = function() {
      return("https://api.anthropic.com/v1/messages")
    },
    
    #' @description
    #' Make API call to Anthropic
    #
    #
    #
    #
    make_api_call = function(chunk_content, model, api_key) {
      # Prepare request body
      body <- list(
        model = model,
        max_tokens = 4096,
        messages = list(
          list(
            role = "user",
            content = chunk_content
          )
        )
      )
      
      self$logger$debug("Sending API request to Anthropic",
                       list(model = model, provider = self$provider_name))
      
      # Make the API request
      response <- httr::POST(
        url = self$get_api_url(),
        httr::add_headers(
          "x-api-key" = api_key,
          "anthropic-version" = "2023-06-01",
          "content-type" = "application/json"
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

        self$logger$error("Anthropic API request failed",
                         list(error = error_message,
                              provider = self$provider_name,
                              model = model,
                              status_code = httr::status_code(response)))

        stop(sprintf("Anthropic API request failed: %s", error_message))
      }
      
      return(response)
    },
    
    #' @description
    #' Extract response content from Anthropic API response
    #
    #
    #
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing Anthropic API response",
                       list(provider = self$provider_name, model = model))
      
      # Parse the response
      content <- httr::content(response, "parsed")
      
      # Check if response has the expected structure
      if (is.null(content) || is.null(content$content) || length(content$content) == 0 ||
          is.null(content$content[[1]]$text)) {
        
        self$logger$error("Unexpected response format from Anthropic API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              content_available = !is.null(content$content),
                              content_count = if(!is.null(content$content)) length(content$content) else 0))
        
        stop("Unexpected response format from Anthropic API")
      }
      
      # Extract the response content
      response_content <- content$content[[1]]$text
      
      return(response_content)
    }
  )
)