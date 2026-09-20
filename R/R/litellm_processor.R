#' LiteLLM Gateway API Processor
#'
#' Concrete implementation of BaseAPIProcessor for a self-hosted LiteLLM
#' gateway. LiteLLM exposes a single OpenAI-compatible API in front of 100+
#' model providers, adding centralized cost tracking, budgets, rate limiting,
#' fallbacks, and load balancing.
#'
#' Models are addressed with a `litellm/` prefix, for example
#' `litellm/gpt-5.5`. The prefix selects this processor; whatever follows it is
#' passed to the gateway untouched, so it can be any name the gateway routes,
#' including one of its own aliases.
#'
#' @export
LiteLLMProcessor <- R6::R6Class("LiteLLMProcessor",
  inherit = BaseAPIProcessor,

  public = list(
    #' @description
    #' Initialize LiteLLM gateway processor
    #' @param base_url Optional custom gateway endpoint
    initialize = function(base_url = NULL) {
      super$initialize("litellm", base_url)
    },

    #' @description
    #' Get default LiteLLM gateway URL. Defaults to a gateway running locally,
    #' which is the common self-hosted setup.
    get_default_api_url = function() {
      return("http://localhost:4000/v1/chat/completions")
    },

    #' @description
    #' Make API call to the LiteLLM gateway
    #' @param chunk_content Prompt text to send
    #' @param model Model identifier, with or without the `litellm/` prefix
    #' @param api_key Gateway master or virtual key. May be empty for a gateway
    #'   started without a master key.
    make_api_call = function(chunk_content, model, api_key) {
      private$post_chat_completions_request(
        chunk_content,
        strip_litellm_model_prefix(model),
        api_key
      )
    },

    #' @description
    #' Extract response content from the gateway response
    #' @param response HTTP response object
    #' @param model Model identifier
    extract_response_content = function(response, model) {
      private$extract_chat_completions_content(response, model)
    }
  )
)

#' Strip the LiteLLM routing prefix from a model name
#'
#' The `litellm/` prefix exists only so mLLMCelltype can route to the gateway
#' processor. The gateway itself knows nothing about it, so it is removed
#' before the request is built.
#'
#' @param model Model identifier, with or without the `litellm/` prefix
#' @return The gateway-facing model name
#' @noRd
strip_litellm_model_prefix <- function(model) {
  sub("^litellm/", "", model, ignore.case = TRUE)
}

#' List models served by a LiteLLM gateway
#'
#' The reachable model set is whatever the gateway operator configured, so it is
#' discovered rather than hardcoded. Useful for checking which models are
#' available before assembling a consensus run.
#'
#' @param api_key Gateway key. Falls back to the `LITELLM_API_KEY` environment
#'   variable, then to an unauthenticated request, which works for a gateway
#'   started without a master key.
#' @param base_url Optional custom gateway endpoint
#' @return Character vector of model ids served by the gateway
#' @export
list_litellm_models <- function(api_key = NULL, base_url = NULL) {
  resolved_key <- if (!is.null(api_key) && nzchar(api_key)) {
    api_key
  } else {
    Sys.getenv("LITELLM_API_KEY", "")
  }

  chat_url <- if (!is.null(base_url) && nzchar(base_url)) {
    base_url
  } else {
    "http://localhost:4000/v1/chat/completions"
  }

  models_url <- sub("/chat/completions$", "/models", chat_url)

  headers <- list("Content-Type" = "application/json")
  if (nzchar(resolved_key)) {
    headers[["Authorization"]] <- paste("Bearer", resolved_key)
  }

  response <- httr::GET(
    url = models_url,
    do.call(httr::add_headers, headers)
  )
  httr::stop_for_status(response)

  parsed <- httr::content(response, as = "parsed", encoding = "UTF-8")
  if (is.null(parsed$data)) {
    stop("Unexpected model list response from LiteLLM gateway")
  }

  ids <- vapply(
    parsed$data,
    function(entry) if (is.null(entry$id)) NA_character_ else as.character(entry$id),
    character(1)
  )
  sort(ids[!is.na(ids)])
}
