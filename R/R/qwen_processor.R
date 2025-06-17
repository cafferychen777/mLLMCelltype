#' Qwen API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Qwen models.
#' Handles Qwen-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
QwenProcessor <- R6::R6Class("QwenProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @field api_url Qwen API endpoint URL
    api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
    
    #' @description
    #' Initialize Qwen processor
    initialize = function() {
      super$initialize("qwen")
    },
    
    #' @description
    #' Make API call to Qwen
    #' @param chunk_content Content for this chunk
    #' @param model Model identifier
    #' @param api_key API key
    #' @return httr response object
    make_api_call = function(chunk_content, model, api_key) {
      # Prepare request body
      body <- list(
        model = model,
        input = list(
          messages = list(
            list(
              role = "user",
              content = chunk_content
            )
          )
        )
      )
      
      self$logger$debug("Sending API request to Qwen",
                       list(model = model, provider = self$provider_name))
      
      # Make the API request
      response <- httr::POST(
        url = self$api_url,
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
      
      # Check if response has the expected structure
      if (is.null(content) || is.null(content$output) || is.null(content$output$choices) || 
          length(content$output$choices) == 0 || is.null(content$output$choices[[1]]$message) ||
          is.null(content$output$choices[[1]]$message$content)) {
        
        self$logger$error("Unexpected response format from Qwen API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              output_available = !is.null(content$output),
                              choices_available = if(!is.null(content$output)) !is.null(content$output$choices) else FALSE,
                              choices_count = if(!is.null(content$output) && !is.null(content$output$choices)) length(content$output$choices) else 0))
        
        stop("Unexpected response format from Qwen API")
      }
      
      # Extract the response content
      response_content <- content$output$choices[[1]]$message$content
      
      return(response_content)
    }
  )
)