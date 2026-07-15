.BUILTIN_PROVIDER_PATTERNS <- c(
  "openai" = "^(gpt-|o[134]|chatgpt-|codex-)",
  "anthropic" = "^claude-",
  "deepseek" = "^deepseek-",
  "gemini" = "^gemini-",
  "qwen" = "^(qwen|qwq-)",
  "stepfun" = "^step-",
  "zhipu" = "^(glm-|chatglm)",
  "minimax" = "^minimax-",
  "grok" = "^grok-",
  "kimi" = "^(kimi-|moonshot)"
)

#' Determine provider from model name
#'
#' This function determines the appropriate provider (e.g., OpenAI, Anthropic, Google, OpenRouter) based on the model name.
#' Uses prefix-based matching for efficient and maintainable provider detection.
#' New models following existing naming conventions are automatically supported.
#'
#' @param model Character string specifying the model name (e.g., "gpt-5.5", "claude-opus-4-7").
#' @return Character string of the provider name (e.g., "openai", "anthropic").
#' @details
#' Supported providers and model prefixes:
#' \itemize{
#'   \item OpenAI: gpt-*, o1*, o3*, o4*, chatgpt-*, codex-* (e.g., 'gpt-5.5', 'gpt-5.4-mini')
#'   \item Anthropic: claude-* (e.g., 'claude-opus-4-7', 'claude-sonnet-4-6')
#'   \item DeepSeek: deepseek-* (e.g., 'deepseek-v4-flash', 'deepseek-v4-pro')
#'   \item Google: gemini-* (e.g., 'gemini-3.1-pro-preview', 'gemini-3-flash-preview')
#'   \item Qwen: qwen*, qwq-* (e.g., 'qwen3.6-plus', 'qwen3.6-flash')
#'   \item Stepfun: step-* (e.g., 'step-3.5-flash', 'step-3')
#'   \item Zhipu: glm-*, chatglm* (e.g., 'glm-5.1', 'glm-5-turbo')
#'   \item MiniMax: minimax-* (e.g., 'MiniMax-M2.7', 'MiniMax-M2.5')
#'   \item Grok: grok-* (e.g., 'grok-4.3', 'grok-4.3-latest')
#'   \item Kimi: kimi-*, moonshot* (e.g., 'kimi-k2.7', 'moonshot-v1-8k')
#'   \item OpenRouter: Any model with '/' in the name (e.g., 'openai/gpt-5.5', 'anthropic/claude-opus-4.7')
#' }
#' @export
get_provider <- function(model) {
  # Normalize model name to lowercase for case-insensitive matching
  model_normalized <- .normalize_required_string(model, "model")
  model_lower <- tolower(model_normalized)

  # OpenRouter models always contain '/' (e.g., 'openai/gpt-5.5')
  if (grepl("/", model_lower)) {
    return("openrouter")
  }

  # Check for custom models
  if (exists(model_lower, envir = custom_models)) {
    model_data <- get(model_lower, envir = custom_models)
    return(model_data$provider)
  }

  # Prefix-based provider detection from the shared built-in registry.
  for (provider in names(.BUILTIN_PROVIDER_PATTERNS)) {
    if (grepl(.BUILTIN_PROVIDER_PATTERNS[[provider]], model_lower)) {
      return(provider)
    }
  }

  # No match — report error with supported prefixes
  supported <- paste(
    vapply(names(.BUILTIN_PROVIDER_PATTERNS), function(p) {
      sprintf("  %s: %s", p, .BUILTIN_PROVIDER_PATTERNS[[p]])
    }, character(1)),
    collapse = "\n"
  )
  stop(
    "Cannot determine provider for model: '", model_normalized, "'\n",
    "Supported model name patterns:\n", supported, "\n",
    "  openrouter: any model containing '/'\n",
    "Tip: Check for typos in the model name."
  )
}
