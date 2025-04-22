#' Get response from a specific model
#' @keywords internal
get_model_response <- function(prompt, model, api_key) {
  # Get the provider for the model
  provider <- get_provider(model)
  
  # First check if it's a custom provider
  if (exists(provider, envir = custom_providers)) {
    return(process_custom(prompt, model, api_key))
  }
  
  # Process with built-in providers
  response <- switch(provider,
    "openai" = process_openai(prompt, model, api_key),
    "anthropic" = process_anthropic(prompt, model, api_key),
    "deepseek" = process_deepseek(prompt, model, api_key),
    "gemini" = process_gemini(prompt, model, api_key),
    "qwen" = process_qwen(prompt, model, api_key),
    "stepfun" = process_stepfun(prompt, model, api_key),
    "zhipu" = process_zhipu(prompt, model, api_key),
    "minimax" = process_minimax(prompt, model, api_key),
    "grok" = process_grok(prompt, model, api_key),
    "openrouter" = process_openrouter(prompt, model, api_key),
    stop("Unsupported model provider: ", provider)
  )
  
  # Log the response (excluding sensitive information)
  write_log(sprintf("Got response from %s model %s", provider, model))
  
  response
}