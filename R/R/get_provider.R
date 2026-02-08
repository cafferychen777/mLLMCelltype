#' Determine provider from model name
#'
#' This function determines the appropriate provider (e.g., OpenAI, Anthropic, Google, OpenRouter) based on the model name.
#' Uses prefix-based matching for efficient and maintainable provider detection.
#' New models following existing naming conventions are automatically supported.
#'
#' @param model Character string specifying the model name (e.g., "gpt-5.2", "claude-sonnet-4.5").
#' @return Character string of the provider name (e.g., "openai", "anthropic").
#' @details
#' Supported providers and model prefixes:
#' \itemize{
#'   \item OpenAI: gpt-*, o1*, o3*, o4*, chatgpt-*, codex-* (e.g., 'gpt-5.2', 'o3-pro', 'o4-mini')
#'   \item Anthropic: claude-* (e.g., 'claude-opus-4.6', 'claude-sonnet-4.5')
#'   \item DeepSeek: deepseek-* (e.g., 'deepseek-chat', 'deepseek-r1')
#'   \item Google: gemini-* (e.g., 'gemini-3-pro', 'gemini-2.5-flash')
#'   \item Qwen: qwen*, qwq-* (e.g., 'qwen3-max', 'qwq-32b')
#'   \item Stepfun: step-* (e.g., 'step-2-mini', 'step-2-16k')
#'   \item Zhipu: glm-*, chatglm* (e.g., 'glm-4.7', 'glm-4-plus')
#'   \item MiniMax: minimax-* (e.g., 'minimax-m2.1', 'minimax-m1')
#'   \item Grok: grok-* (e.g., 'grok-4', 'grok-4-heavy')
#'   \item OpenRouter: Any model with '/' in the name (e.g., 'openai/gpt-5.2', 'anthropic/claude-sonnet-4.5')
#' }
#' @export
get_provider <- function(model) {
  if (!is.character(model) || length(model) != 1 || is.na(model) || !nzchar(trimws(model))) {
    stop("model must be a non-empty character scalar")
  }

  # Normalize model name to lowercase for case-insensitive matching
  model_lower <- tolower(model)

  # OpenRouter models always contain '/' (e.g., 'openai/gpt-5.2')
  if (grepl("/", model_lower)) {
    return("openrouter")
  }

  # Check for custom models
  if (exists(model_lower, envir = custom_models)) {
    model_data <- get(model_lower, envir = custom_models)
    return(model_data$provider)
  }

  # Prefix-based provider detection
  # Each regex matches the naming convention of a provider's models
  provider_patterns <- list(
    "openai"    = "^(gpt-|o[0-9]|chatgpt-|codex-)",
    "anthropic" = "^claude-",
    "deepseek"  = "^deepseek-",
    "gemini"    = "^gemini-",
    "qwen"      = "^(qwen|qwq-)",
    "stepfun"   = "^step-",
    "zhipu"     = "^(glm-|chatglm)",
    "minimax"   = "^minimax-",
    "grok"      = "^grok-"
  )

  for (provider in names(provider_patterns)) {
    if (grepl(provider_patterns[[provider]], model_lower)) {
      return(provider)
    }
  }

  # No match — report error with supported prefixes
  supported <- paste(
    vapply(names(provider_patterns), function(p) {
      sprintf("  %s: %s", p, provider_patterns[[p]])
    }, character(1)),
    collapse = "\n"
  )
  stop(
    "Cannot determine provider for model: '", model, "'\n",
    "Supported model name patterns:\n", supported, "\n",
    "  openrouter: any model containing '/'\n",
    "Tip: Check for typos in the model name."
  )
}
