#' Get response from a specific model
#' @keywords internal
get_model_response <- function(prompt, model, api_key) {
  # Get the provider for the model
  provider <- get_provider(model)
  
  log_debug("Starting model request", list(
    provider = provider,
    model = model,
    prompt_length = nchar(prompt)
  ))
  
  # First check if it's a custom provider
  if (exists(provider, envir = custom_providers)) {
    log_debug("Using custom provider", list(provider = provider))
    return(process_custom(prompt, model, api_key))
  }
  
  # Track timing for API call
  start_time <- Sys.time()
  
  tryCatch({
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
    
    # Calculate duration and log successful API call
    duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    get_logger()$log_api_call(provider, model, duration, TRUE)
    
    log_info("Model response received successfully", list(
      provider = provider,
      model = model,
      response_length = nchar(response),
      duration_seconds = round(duration, 3)
    ))
    
    return(response)
    
  }, error = function(e) {
    # Calculate duration and log failed API call
    duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    get_logger()$log_api_call(provider, model, duration, FALSE)
    
    log_error("Model request failed", list(
      provider = provider,
      model = model,
      error = e$message,
      duration_seconds = round(duration, 3)
    ))
    
    stop(e)
  })
}