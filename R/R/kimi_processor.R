#' Kimi API Processor
#'
#' Concrete implementation of BaseAPIProcessor for Kimi models. By default it
#' targets the Moonshot AI Open Platform over the OpenAI-compatible Chat
#' Completions protocol, with k2 thinking mode disabled for deterministic
#' output. A custom `base_url` may instead point at the Kimi Code platform
#' (api.kimi.com/coding), which speaks both protocols; the protocol is
#' inferred from the effective endpoint URL. URLs ending in '/messages' use
#' the Anthropic Messages protocol; the Kimi Code base
#' 'https://api.kimi.com/coding' and URLs ending in '/chat/completions' use
#' OpenAI-compatible Chat Completions.
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
      private$endpoint <- private$resolve_kimi_endpoint(self$get_api_url())
    },

    #' @description
    #' Get default Kimi API URL
    #
    get_default_api_url = function() {
      return("https://api.moonshot.cn/v1/chat/completions")
    },

    #' @description
    #' Make API call to Kimi
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier (e.g., 'kimi-k2.6', 'moonshot-v1-8k')
    #' @param api_key Moonshot API key
    make_api_call = function(chunk_content, model, api_key) {
      if (identical(private$endpoint$mode, "anthropic")) {
        return(private$post_messages_request(chunk_content, model, api_key))
      }
      private$post_chat_completions_request(
        chunk_content,
        model,
        api_key,
        api_url = private$endpoint$url,
        body_extra = list(
          temperature = 0.6,
          max_tokens = 4096,
          thinking = list(type = "disabled")
        )
      )
    },

    #' @description
    #' Extract response content from Kimi API response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      if (identical(private$endpoint$mode, "anthropic")) {
        return(private$extract_messages_content(response, model, "Kimi"))
      }
      private$extract_chat_completions_content(response, model)
    },

    #' @description
    #' Extract normalized token usage from a Kimi API response
    #' @param response HTTP response object
    extract_usage = function(response) {
      if (identical(private$endpoint$mode, "anthropic")) {
        return(private$extract_usage_fields(
          response,
          prompt_field = "input_tokens",
          completion_field = "output_tokens",
          total_field = NULL,
          cost_field = NULL,
          derive_total = TRUE
        ))
      }
      private$extract_usage_fields(response)
    }
  ),

  private = list(
    # Resolved endpoint contract: list(url = <final URL>, mode = "openai"|"anthropic")
    endpoint = NULL,

    # Classify the effective endpoint URL into its final form and protocol.
    # Kimi Code base URLs are completed to their canonical endpoints; every
    # other URL is used as-is in OpenAI-compatible mode.
    resolve_kimi_endpoint = function(url) {
      lower <- tolower(url)
      if (grepl("/messages$", lower)) {
        return(list(url = url, mode = "anthropic"))
      }
      if (grepl("/chat/completions$", lower)) {
        return(list(url = url, mode = "openai"))
      }
      if (grepl("/coding/v1$", lower)) {
        completed <- paste0(url, "/chat/completions")
        self$logger$debug(
          sprintf("Using Kimi Code OpenAI-compatible endpoint: %s", completed),
          list(provider = self$provider_name)
        )
        return(list(url = completed, mode = "openai"))
      }
      if (grepl("/coding$", lower)) {
        completed <- paste0(url, "/v1/chat/completions")
        self$logger$debug(
          sprintf("Using Kimi Code OpenAI-compatible endpoint: %s", completed),
          list(provider = self$provider_name)
        )
        return(list(url = completed, mode = "openai"))
      }
      list(url = url, mode = "openai")
    },

    # Anthropic Messages protocol request (Kimi Code endpoints)
    post_messages_request = function(chunk_content, model, api_key) {
      body <- list(
        model = model,
        max_tokens = 4096,
        temperature = 0.6,
        messages = list(
          list(
            role = "user",
            content = chunk_content
          )
        )
      )

      self$logger$debug("Sending API request to Kimi (Anthropic-compatible mode)",
                       list(model = model, provider = self$provider_name))

      response <- httr::POST(
        url = private$endpoint$url,
        httr::add_headers(
          "x-api-key" = api_key,
          "anthropic-version" = "2023-06-01",
          "content-type" = "application/json"
        ),
        body = body,
        encode = "json",
        httr::timeout(get_api_request_timeout())
      )

      private$stop_for_http_error(response, model)
      response
    }
  )
)
