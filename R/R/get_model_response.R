#' Get response from a specific model
#' @keywords internal
get_model_response <- function(prompt, model, api_key, base_urls = NULL) {
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

  # First check if it's a custom provider
  if (exists(provider, envir = custom_providers)) {
    log_debug("Using custom provider", list(provider = provider))
    return(process_custom(prompt, model, api_key))
  }

  # Delegate to provider-specific processor (timing & logging handled by BaseAPIProcessor)
  switch(provider,
    "openai" = process_openai(prompt, model, api_key, provider_base_url),
    "anthropic" = process_anthropic(prompt, model, api_key, provider_base_url),
    "deepseek" = process_deepseek(prompt, model, api_key, provider_base_url),
    "gemini" = process_gemini(prompt, model, api_key, provider_base_url),
    "qwen" = process_qwen(prompt, model, api_key, provider_base_url),
    "stepfun" = process_stepfun(prompt, model, api_key, provider_base_url),
    "zhipu" = process_zhipu(prompt, model, api_key, provider_base_url),
    "minimax" = process_minimax(prompt, model, api_key, provider_base_url),
    "grok" = process_grok(prompt, model, api_key, provider_base_url),
    "openrouter" = process_openrouter(prompt, model, api_key, provider_base_url),
    stop("Unsupported model provider: ", provider)
  )
}