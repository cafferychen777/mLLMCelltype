#' DeepSeek API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for DeepSeek models.
#' Handles DeepSeek-specific API calls, authentication, and response parsing.
#'
#' @export
DeepSeekProcessor <- R6::R6Class("DeepSeekProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize DeepSeek processor
    #' @param base_url Optional custom API endpoint
    initialize = function(base_url = NULL) {
      super$initialize("deepseek", base_url)
    },

    #' @description
    #' Get default DeepSeek API URL
    #
    get_default_api_url = function() {
      return("https://api.deepseek.com/v1/chat/completions")
    },
    
    #' @description
    #' Make API call to DeepSeek
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key DeepSeek API key
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(
        chunk_content,
        model,
        api_key,
        body_extra = list(
          temperature = 0.7,
          max_tokens = 4096,
          stream = FALSE,
          thinking = list(type = "disabled")
        )
      )
    },
    
    #' @description
    #' Extract response content from DeepSeek API response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      private$extract_chat_completions_content(response, model)
    }
  )
)
