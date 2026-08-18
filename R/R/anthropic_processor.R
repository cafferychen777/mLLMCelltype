#' Anthropic API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Anthropic models.
#' Handles Anthropic-specific API calls, authentication, and response parsing.
#'
#' @export
AnthropicProcessor <- R6::R6Class("AnthropicProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize Anthropic processor
    #' @param base_url Optional custom API endpoint
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
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key Anthropic API key
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
        body = body,
        encode = "json",
        httr::timeout(get_api_request_timeout())
      )
      
      private$stop_for_http_error(response, model, "Anthropic")
      
      return(response)
    },
    
    #' @description
    #' Extract response content from Anthropic API response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      private$extract_messages_content(response, model, "Anthropic")
    },

    #' @description
    #' Extract normalized Anthropic token usage
    #' @param response HTTP response object
    extract_usage = function(response) {
      private$extract_usage_fields(
        response,
        prompt_field = "input_tokens",
        completion_field = "output_tokens",
        total_field = NULL,
        cost_field = NULL,
        derive_total = TRUE
      )
    }
  )
)
