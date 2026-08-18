#' Gemini API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Gemini models.
#' Handles Gemini-specific API calls, authentication, and response parsing.
#'
#' @export
GeminiProcessor <- R6::R6Class("GeminiProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize Gemini processor
    #' @param base_url Optional custom API endpoint
    initialize = function(base_url = NULL) {
      super$initialize("gemini", base_url)
    },

    #' @description
    #' Get default Gemini API URL template
    #
    get_default_api_url = function() {
      return("https://generativelanguage.googleapis.com/v1beta/models")
    },

    #' @description
    #' Get API URL for specific model
    #' @param model Model identifier
    get_api_url_for_model = function(model) {
      return(paste0(self$get_api_url(), "/", model, ":generateContent"))
    },
    
    #' @description
    #' Make API call to Gemini
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key Gemini API key
    make_api_call = function(chunk_content, model, api_key) {
      # Build API URL with model
      api_url <- self$get_api_url_for_model(model)
      
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
        body = body,
        encode = "json",
        httr::timeout(get_api_request_timeout())
      )
      
      private$stop_for_http_error(response, model, "Gemini")
      
      return(response)
    },
    
    #' @description
    #' Extract response content from Gemini API response
    #' @param response HTTP response object
    #' @param model Model identifier
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
    },

    #' @description
    #' Extract normalized Gemini token usage
    #' @param response HTTP response object
    extract_usage = function(response) {
      private$extract_usage_fields(
        response,
        usage_field = "usageMetadata",
        prompt_field = "promptTokenCount",
        completion_field = "candidatesTokenCount",
        total_field = "totalTokenCount",
        cost_field = NULL
      )
    }
  )
)
