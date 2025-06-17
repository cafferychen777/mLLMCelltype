#' Gemini API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Gemini models.
#' Handles Gemini-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
GeminiProcessor <- R6::R6Class("GeminiProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @field api_url_template Gemini API endpoint URL template
    api_url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    
    #' @description
    #' Initialize Gemini processor
    initialize = function() {
      super$initialize("gemini")
    },
    
    #' @description
    #' Make API call to Gemini
    #' @param chunk_content Content for this chunk
    #' @param model Model identifier
    #' @param api_key API key
    #' @return httr response object
    make_api_call = function(chunk_content, model, api_key) {
      # Build API URL with model
      api_url <- gsub("\\{model\\}", model, self$api_url_template)
      
      # Prepare request body
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
      
      self$logger$debug("Sending API request to Gemini",
                       list(model = model, provider = self$provider_name))
      
      # Make the API request
      response <- httr::POST(
        url = api_url,
        httr::add_headers(
          "x-goog-api-key" = api_key,
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
        
        self$logger$error("Gemini API request failed",
                         list(error = error_message,
                              provider = self$provider_name,
                              model = model,
                              status_code = httr::status_code(response)))
        
        stop(sprintf("Gemini API request failed: %s", error_message))
      }
      
      return(response)
    },
    
    #' @description
    #' Extract response content from Gemini API response
    #' @param response httr response object
    #' @param model Model identifier
    #' @return Extracted text content
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing Gemini API response",
                       list(provider = self$provider_name, model = model))
      
      # Parse the response
      content <- httr::content(response, "parsed")
      
      # Check if response has the expected structure
      if (is.null(content) || is.null(content$candidates) || length(content$candidates) == 0 ||
          is.null(content$candidates[[1]]$content) || is.null(content$candidates[[1]]$content$parts) ||
          length(content$candidates[[1]]$content$parts) == 0 || is.null(content$candidates[[1]]$content$parts[[1]]$text)) {
        
        self$logger$error("Unexpected response format from Gemini API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              candidates_available = !is.null(content$candidates),
                              candidates_count = if(!is.null(content$candidates)) length(content$candidates) else 0))
        
        stop("Unexpected response format from Gemini API")
      }
      
      # Extract the response content
      response_content <- content$candidates[[1]]$content$parts[[1]]$text
      
      return(response_content)
    }
  )
)