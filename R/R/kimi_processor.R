#' Kimi API Processor
#'
#' Concrete implementation of BaseAPIProcessor for Kimi (Moonshot AI) models.
#' Kimi exposes an Anthropic-compatible Messages API: it uses the same
#' `/v1/messages` endpoint, `x-api-key` + `anthropic-version` headers, and
#' returns content under `content[[1]]$text`.
#'
#' @export
KimiProcessor <- R6::R6Class("KimiProcessor",
  inherit = BaseAPIProcessor,

  public = list(
    #' @description
    #' Initialize Kimi processor
    #' @param base_url Optional custom API endpoint
    initialize = function(base_url = NULL) {
      super$initialize("kimi", base_url)
    },

    #' @description
    #' Get default Kimi API URL
    #
    get_default_api_url = function() {
      return("https://api.kimi.com/coding/v1/messages")
    },

    #' @description
    #' Make API call to Kimi
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier
    #' @param api_key Kimi API key
    make_api_call = function(chunk_content, model, api_key) {
      # Prepare request body (Anthropic Messages format)
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

      self$logger$debug("Sending API request to Kimi",
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
        httr::timeout(30)
      )

      private$stop_for_http_error(response, model, "Kimi")

      return(response)
    },

    #' @description
    #' Extract response content from Kimi API response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      self$logger$debug("Parsing Kimi API response",
                       list(provider = self$provider_name, model = model))

      # Parse the response
      content <- httr::content(response, "parsed")

      # Check if response has the expected structure (Anthropic format)
      if (is.null(content) || is.null(content$content) || length(content$content) == 0 ||
          is.null(content$content[[1]]$text)) {

        self$logger$error("Unexpected response format from Kimi API",
                         list(provider = self$provider_name,
                              model = model,
                              content_structure = names(content),
                              content_available = !is.null(content$content),
                              content_count = if(!is.null(content$content)) length(content$content) else 0))

        stop("Unexpected response format from Kimi API")
      }

      # Extract the response content
      response_content <- content$content[[1]]$text

      return(response_content)
    },

    #' @description
    #' Extract normalized Kimi token usage
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
