.BUILTIN_PROVIDER_SPECS <- list(
  openai = list(
    pattern = "^(gpt-|o[134]|chatgpt-|codex-)",
    processor_class = "OpenAIProcessor",
    display_name = "OpenAI"
  ),
  anthropic = list(
    pattern = "^claude-",
    processor_class = "AnthropicProcessor",
    display_name = "Anthropic"
  ),
  deepseek = list(
    pattern = "^deepseek-",
    processor_class = "DeepSeekProcessor",
    display_name = "DeepSeek"
  ),
  gemini = list(
    pattern = "^gemini-",
    processor_class = "GeminiProcessor",
    display_name = "Gemini",
    api_key_env_aliases = "GOOGLE_API_KEY"
  ),
  qwen = list(
    pattern = "^(qwen|qwq-)",
    processor_class = "QwenProcessor",
    display_name = "Qwen"
  ),
  stepfun = list(
    pattern = "^step-",
    processor_class = "StepFunProcessor",
    display_name = "StepFun"
  ),
  zhipu = list(
    pattern = "^(glm-|chatglm)",
    processor_class = "ZhipuProcessor",
    display_name = "Zhipu"
  ),
  minimax = list(
    pattern = "^minimax-",
    processor_class = "MinimaxProcessor",
    display_name = "MiniMax"
  ),
  grok = list(
    pattern = "^grok-",
    processor_class = "GrokProcessor",
    display_name = "Grok"
  ),
  kimi = list(
    pattern = "^(kimi-|moonshot-)",
    processor_class = "KimiProcessor",
    display_name = "Kimi",
    api_key_env_aliases = "MOONSHOT_API_KEY"
  ),
  litellm = list(
    pattern = NULL,
    processor_class = "LiteLLMProcessor",
    display_name = "LiteLLM"
  ),
  openrouter = list(
    pattern = NULL,
    processor_class = "OpenRouterProcessor",
    display_name = "OpenRouter"
  )
)

# Derived compatibility view used by provider detection and error reporting.
.BUILTIN_PROVIDER_PATTERNS <- vapply(
  .BUILTIN_PROVIDER_SPECS[
    vapply(.BUILTIN_PROVIDER_SPECS, function(spec) !is.null(spec$pattern), logical(1))
  ],
  function(spec) spec$pattern,
  character(1)
)

get_builtin_provider_spec <- function(provider) {
  provider <- tolower(.normalize_required_string(provider, "provider"))
  spec <- .BUILTIN_PROVIDER_SPECS[[provider]]
  if (is.null(spec)) {
    stop("Unsupported model provider: ", provider)
  }
  spec
}

get_builtin_provider_display_name <- function(provider) {
  provider <- tolower(.normalize_required_string(provider, "provider"))
  spec <- .BUILTIN_PROVIDER_SPECS[[provider]]
  if (is.null(spec)) provider else spec$display_name
}

resolve_builtin_processor_class <- function(provider) {
  spec <- get_builtin_provider_spec(provider)
  class_name <- spec$processor_class
  valid_class_name <- is.character(class_name) &&
    length(class_name) == 1 &&
    !is.na(class_name) &&
    nzchar(class_name)
  if (!valid_class_name || !exists(class_name, envir = environment(), inherits = TRUE)) {
    stop("Invalid processor class for built-in provider: ", provider)
  }
  processor_class <- get(
    class_name,
    envir = environment(),
    inherits = TRUE
  )
  if (!inherits(processor_class, "R6ClassGenerator")) {
    stop("Invalid processor class for built-in provider: ", provider)
  }
  processor_class
}

new_builtin_provider_processor <- function(provider, base_url = NULL) {
  resolve_builtin_processor_class(provider)$new(base_url = base_url)
}

validate_builtin_provider_registry <- function() {
  required_fields <- c("processor_class", "display_name")
  for (provider in names(.BUILTIN_PROVIDER_SPECS)) {
    spec <- .BUILTIN_PROVIDER_SPECS[[provider]]
    if (!is.list(spec) || !all(required_fields %in% names(spec))) {
      stop("Invalid built-in provider specification: ", provider)
    }
    for (field in required_fields) {
      value <- spec[[field]]
      if (!is.character(value) || length(value) != 1 || is.na(value) || !nzchar(value)) {
        stop("Invalid ", field, " for built-in provider: ", provider)
      }
    }
    if (!is.null(spec$pattern)) {
      valid_pattern <- is.character(spec$pattern) &&
        length(spec$pattern) == 1 &&
        !is.na(spec$pattern) &&
        nzchar(spec$pattern)
      if (!valid_pattern) {
        stop("Invalid pattern for built-in provider: ", provider)
      }
      tryCatch(
        grepl(spec$pattern, ""),
        error = function(e) {
          stop("Invalid pattern for built-in provider: ", provider)
        }
      )
    }
    if (!is.null(spec$api_key_env_aliases)) {
      aliases <- spec$api_key_env_aliases
      valid_aliases <- is.character(aliases) &&
        length(aliases) > 0 &&
        !anyNA(aliases) &&
        all(grepl("^[A-Za-z_][A-Za-z0-9_]*$", aliases)) &&
        !anyDuplicated(aliases)
      if (!valid_aliases) {
        stop("Invalid API key environment aliases for built-in provider: ", provider)
      }
    }
    resolve_builtin_processor_class(provider)
  }
  invisible(TRUE)
}

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
#'   \item Kimi (Moonshot AI Open Platform): kimi-*, moonshot-* (e.g., 'kimi-k2.6', 'moonshot-v1-8k')
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
