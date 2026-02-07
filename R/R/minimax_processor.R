#' Minimax API Processor
#' 
#' Concrete implementation of BaseAPIProcessor for Minimax models.
#' Handles Minimax-specific API calls, authentication, and response parsing.
#'
#' @importFrom R6 R6Class
#' @export
MinimaxProcessor <- R6::R6Class("MinimaxProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize Minimax processor
    #
    initialize = function(base_url = NULL) {
      super$initialize("minimax", base_url)
    },

    #' @description
    #' Get default Minimax API URL
    #
    get_default_api_url = function() {
      return("https://api.minimax.chat/v1/text/chatcompletion_v2")
    },
    
    #' @description
    #' Make API call to Minimax
    #
    #
    #
    #
    make_api_call = function(chunk_content, model, api_key) {
      # Prepare request body
      body <- list(
        model = model,
        messages = list(
          list(
            role = "user",
            content = chunk_content
          )
        )
      )
      
      self$logger$debug("Sending API request to Minimax",
                       list(model = model, provider = self$provider_name))
      
      # Make the API request
      response <- httr::POST(
        url = self$get_api_url(),
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
        
        self$logger$error("Minimax API request failed",
                         list(error = error_message,
                              provider = self$provider_name,
                              model = model,
                              status_code = httr::status_code(response)))
        
        stop(sprintf("Minimax API request failed: %s", error_message))
      }
      
      return(response)
    },
    
    #' @description
    #' Extract response content from Minimax API response
    #
    #
    #
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing Minimax API response",
                       list(provider = self$provider_name, model = model))
      
      # Parse the response
      content <- httr::content(response, "parsed")
      
      # Check if response has the expected structure
      if (is.null(content) || is.null(content$choices) || length(content$choices) == 0 ||
          is.null(content$choices[[1]]$messages) || length(content$choices[[1]]$messages) == 0 ||
          is.null(content$choices[[1]]$messages[[1]]$text)) {
        
        self$logger$error("Unexpected response format from Minimax API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              choices_available = !is.null(content$choices),
                              choices_count = if(!is.null(content$choices)) length(content$choices) else 0))
        
        stop("Unexpected response format from Minimax API")
      }
      
      # Extract the response content
      response_content <- content$choices[[1]]$messages[[1]]$text
      
      return(response_content)
    }
  )
)