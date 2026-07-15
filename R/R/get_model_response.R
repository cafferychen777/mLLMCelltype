# Return the built-in provider dispatch table.
get_builtin_provider_processors <- function() {
  list(
    openai = process_openai,
    anthropic = process_anthropic,
    deepseek = process_deepseek,
    gemini = process_gemini,
    qwen = process_qwen,
    stepfun = process_stepfun,
    zhipu = process_zhipu,
    minimax = process_minimax,
    grok = process_grok,
    kimi = process_kimi,
    openrouter = process_openrouter
  )
}

.MODEL_REQUEST_RETRY <- list(
  MAX_ATTEMPTS = 3L,
  BASE_DELAY_SECONDS = 5
)

wait_before_model_retry <- function(seconds) {
  Sys.sleep(seconds)
}

dispatch_model_request_once <- function(prompt, model, api_key, provider,
                                        provider_base_url) {
  if (exists(provider, envir = custom_providers)) {
    log_debug("Using custom provider", list(provider = provider))
    response <- if (is.null(provider_base_url)) {
      process_custom(prompt, model, api_key)
    } else {
      process_custom(prompt, model, api_key, provider_base_url)
    }
  } else {
    processors <- get_builtin_provider_processors()
    configured_providers <- c(names(.BUILTIN_PROVIDER_PATTERNS), "openrouter")
    if (!setequal(names(processors), configured_providers)) {
      stop("Built-in provider registry mismatch")
    }
    processor <- processors[[provider]]
    if (is.null(processor)) {
      stop("Unsupported model provider: ", provider)
    }
    response <- processor(prompt, model, api_key, provider_base_url)
  }

  tryCatch(
    normalize_model_response_lines(response),
    error = function(e) {
      stop("Invalid response format from provider '", provider, "'")
    }
  )
}

#' Get response from a specific model
#'
#' @param prompt Non-empty prompt string
#' @param model Non-empty model name
#' @param api_key Non-empty API key
#' @param base_urls Optional shared or provider-specific base URL configuration
#' @return Provider response as a character vector
#' @keywords internal
get_model_response <- function(prompt, model, api_key, base_urls = NULL) {
  prompt <- .normalize_required_string(prompt, "prompt")
  model <- .normalize_required_string(model, "model")
  api_key <- .normalize_required_string(api_key, "api_key")

  # Get the provider for the model
  provider <- get_provider(model)

  # Resolve provider-specific base URL (same pattern as annotate_cell_types)
  provider_base_url <- resolve_provider_base_url(provider, base_urls)

  log_debug("Starting model request", list(
    provider = provider,
    model = model,
    prompt_length = nchar(prompt),
    custom_url = !is.null(provider_base_url)
  ))

  for (attempt in seq_len(.MODEL_REQUEST_RETRY$MAX_ATTEMPTS)) {
    result <- tryCatch(
      list(
        success = TRUE,
        response = dispatch_model_request_once(
          prompt,
          model,
          api_key,
          provider,
          provider_base_url
        )
      ),
      error = function(error) list(success = FALSE, error = error)
    )

    if (isTRUE(result$success)) {
      return(result$response)
    }

    retryable <- inherits(result$error, "mllm_api_error") &&
      isTRUE(result$error$retryable)
    if (!retryable || attempt == .MODEL_REQUEST_RETRY$MAX_ATTEMPTS) {
      stop(result$error)
    }

    wait_seconds <- .MODEL_REQUEST_RETRY$BASE_DELAY_SECONDS * 2^(attempt - 1)
    get_logger()$warn("Retrying transient model request failure", list(
      provider = provider,
      model = model,
      attempt = attempt,
      max_attempts = .MODEL_REQUEST_RETRY$MAX_ATTEMPTS,
      wait_seconds = wait_seconds,
      error = result$error$message
    ))
    wait_before_model_retry(wait_seconds)
  }

  stop("Model request retry loop ended unexpectedly")
}
