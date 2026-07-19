# MiniMax reports business errors via HTTP 200 + base_resp$status_code. The
# codes below are transient per the MiniMax docs (1000 unknown error, 1001
# timeout, 1002 rate limit, 1039 token limit) and should be retried; all other
# non-zero codes (1004 auth, 1008 balance, 1027 content policy, 2013 invalid
# params) are fatal.
.MINIMAX_TRANSIENT_STATUS_CODES <- c(1000L, 1001L, 1002L, 1039L)

#' Minimax API Processor
#'
#' Concrete implementation of BaseAPIProcessor for Minimax models.
#' Handles Minimax-specific API calls, authentication, and response parsing.
#'
#' @export
MinimaxProcessor <- R6::R6Class("MinimaxProcessor",
  inherit = BaseAPIProcessor,
  
  public = list(
    #' @description
    #' Initialize Minimax processor
    #' @param base_url Optional custom API endpoint
    initialize = function(base_url = NULL) {
      super$initialize("minimax", base_url)
    },

    #' @description
    #' Get default MiniMax OpenAI-compatible chat completions API URL
    #
    get_default_api_url = function() {
      return("https://api.minimax.io/v1/chat/completions")
    },
    
    #' @description
    #' Make API call to Minimax
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key MiniMax API key
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(chunk_content, model, api_key)
    },
    
    #' @description
    #' Extract response content from Minimax API response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing MiniMax API response",
                       list(provider = self$provider_name, model = model))
      
      # Parse the response
      content <- httr::content(response, "parsed")
      
      base_resp <- content$base_resp
      if (is.list(base_resp) &&
          !is.null(base_resp$status_code) &&
          !identical(as.integer(base_resp$status_code), 0L)) {
        status_msg_available <- is.character(base_resp$status_msg) &&
          length(base_resp$status_msg) == 1 &&
          !is.na(base_resp$status_msg) &&
          nzchar(base_resp$status_msg)
        error_message <- if (status_msg_available) {
          base_resp$status_msg
        } else {
          "Unknown MiniMax API error"
        }
        status_code_int <- as.integer(base_resp$status_code)
        stop_api_request_error(
          sprintf("MiniMax API error: %s", error_message),
          retryable = status_code_int %in% .MINIMAX_TRANSIENT_STATUS_CODES
        )
      }

      if (!is.null(content$choices) &&
          length(content$choices) > 0 &&
          !is.null(content$choices[[1]]$message$content)) {
        response_content <- content$choices[[1]]$message$content
      } else if (!is.null(content$choices) &&
                 length(content$choices) > 0 &&
                 !is.null(content$choices[[1]]$messages) &&
                 length(content$choices[[1]]$messages) > 0 &&
                 !is.null(content$choices[[1]]$messages[[1]]$text)) {
        # Legacy MiniMax text/chatcompletion_v2 shape.
        response_content <- content$choices[[1]]$messages[[1]]$text
      } else {
        
        self$logger$error("Unexpected response format from MiniMax API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              choices_available = !is.null(content$choices),
                              choices_count = if (!is.null(content$choices)) {
                                length(content$choices)
                              } else {
                                0
                              }))
        
        stop("Unexpected response format from MiniMax API")
      }

      response_content <- gsub(
        "<think>[\\s\\S]*?</think>\\s*",
        "",
        response_content,
        perl = TRUE
      )

      return(response_content)
    }
  )
)
