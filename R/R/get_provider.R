# Define global variables
utils::globalVariables(c("custom_models"))

#' Determine provider from model name
#'
#' This function determines the appropriate provider (e.g., OpenAI, Anthropic, Google) based on the model name.
#'
#' @param model Character string specifying the model name to check
#' @return Character string with the provider name
#' @export
get_provider <- function(model) {
  # Normalize model name to lowercase for comparison
  model <- tolower(model)
  
  # List of supported models for each provider (all in lowercase)
  openai_models <- c("gpt-4o", "o1")
  anthropic_models <- c("claude-3-7-sonnet-20250219", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus")
  deepseek_models <- c("deepseek-chat", "deepseek-reasoner")
  gemini_models <- c("gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash")
  qwen_models <- c("qwen-max-2025-01-25")
  stepfun_models <- c("step-2-mini", "step-2-16k", "step-1-8k")
  zhipu_models <- c("glm-4-plus", "glm-3-turbo")
  minimax_models <- c("minimax-text-01")
  grok_models <- c("grok-3", "grok-3-latest", "grok-3-fast", "grok-3-fast-latest", "grok-3-mini", "grok-3-mini-latest", "grok-3-mini-fast", "grok-3-mini-fast-latest")
  
  # Check for custom models first
  if (exists(model, envir = custom_models)) {
    model_data <- get(model, envir = custom_models)
    return(model_data$provider)
  }
  
  # Determine provider based on model name for built-in providers
  if (model %in% openai_models) {
    return("openai")
  } else if (model %in% anthropic_models) {
    return("anthropic")
  } else if (model %in% deepseek_models) {
    return("deepseek")
  } else if (model %in% gemini_models) {
    return("gemini")
  } else if (model %in% qwen_models) {
    return("qwen")
  } else if (model %in% stepfun_models) {
    return("stepfun")
  } else if (model %in% zhipu_models) {
    return("zhipu")
  } else if (model %in% minimax_models) {
    return("minimax")
  } else if (model %in% grok_models) {
    return("grok")
  }
  
  # Get list of all supported models
  all_models <- c(
    openai_models, anthropic_models, deepseek_models,
    gemini_models, qwen_models, stepfun_models, zhipu_models, minimax_models, grok_models
  )
  
  # Add custom models to the list
  custom_model_names <- ls(envir = custom_models)
  if (length(custom_model_names) > 0) {
    all_models <- c(all_models, custom_model_names)
  }
  
  # If model not found in any provider's list
  stop("Unsupported model: ", model, "\n",
       "Supported models are: ", 
       paste(all_models, collapse = ", "))
}