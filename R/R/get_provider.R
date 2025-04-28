# Define global variables
utils::globalVariables(c("custom_models"))

#' Determine provider from model name
#'
#' This function determines the appropriate provider (e.g., OpenAI, Anthropic, Google, OpenRouter) based on the model name.
#'
#' @param model Character string specifying the model name to check
#' @return Character string with the provider name
#' @details
#' Supported providers and models include:
#' \itemize{
#'   \item OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
#'   \item Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
#'   \item DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
#'   \item Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
#'   \item Qwen: 'qwen-max-2025-01-25'
#'   \item Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
#'   \item Zhipu: 'glm-4-plus', 'glm-3-turbo'
#'   \item MiniMax: 'minimax-text-01'
#'   \item Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
#'   \item OpenRouter: Provides access to models from multiple providers through a single API. Format: 'provider/model-name'
#'     \itemize{
#'       \item OpenAI models: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#'       \item Anthropic models: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#'       \item Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#'       \item Google models: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#'       \item Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#'       \item Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'
#'     }
#' }
#' @importFrom utils adist
#' @export
get_provider <- function(model) {
  # Normalize model name to lowercase for comparison
  model <- tolower(model)

  # Special case for OpenRouter models which may contain '/' in the model name
  if (grepl("/", model)) {
    # OpenRouter models are in the format 'provider/model'
    # e.g., 'anthropic/claude-3-opus', 'google/gemini-2.5-pro-preview-03-25'
    return("openrouter")
  }

  # List of supported models for each provider (all in lowercase)
  openai_models <- c("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini", "o1-preview", "o1-pro")
  anthropic_models <- c("claude-3-7-sonnet-20250219", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus")
  deepseek_models <- c("deepseek-chat", "deepseek-reasoner")
  gemini_models <- c("gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash")
  qwen_models <- c("qwen-max-2025-01-25")
  stepfun_models <- c("step-2-mini", "step-2-16k", "step-1-8k")
  zhipu_models <- c("glm-4-plus", "glm-3-turbo")
  minimax_models <- c("minimax-text-01")
  grok_models <- c("grok-3", "grok-3-latest", "grok-3-fast", "grok-3-fast-latest", "grok-3-mini", "grok-3-mini-latest", "grok-3-mini-fast", "grok-3-mini-fast-latest")
  openrouter_models <- c(
    # OpenAI models
    "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4-turbo", "openai/gpt-4", "openai/gpt-3.5-turbo",
    "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "openai/o1", "openai/o1-mini", "openai/o1-preview", "openai/o1-pro",
    "openai/o4-mini",

    # Anthropic models
    "anthropic/claude-3-7-sonnet-20250219", "anthropic/claude-3-5-sonnet-latest", "anthropic/claude-3-5-haiku-latest", "anthropic/claude-3-opus",
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-haiku", "anthropic/claude-3.7-sonnet",

    # Meta models
    "meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-8b-instruct", "meta-llama/llama-2-70b-chat",
    "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free", "meta-llama/llama-3.2-1b-instruct:free", "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-4-maverick:free", "meta-llama/llama-4-scout:free",

    # Google models
    "google/gemini-2.5-pro-preview-03-25", "google/gemini-1.5-pro-latest", "google/gemini-1.5-flash",
    "google/gemini-2.0-flash-001", "google/gemini-2.0-flash-exp:free", "google/gemini-2.5-flash-preview",
    "google/gemma-2-9b-it:free", "google/gemma-3-12b-it:free", "google/gemma-3-1b-it:free", "google/gemma-3-27b-it:free", "google/gemma-3-4b-it:free",
    "google/learnlm-1.5-pro-experimental:free",

    # Mistral models
    "mistralai/mistral-large", "mistralai/mistral-medium", "mistralai/mistral-small",
    "mistralai/mistral-7b-instruct:free", "mistralai/mistral-small-24b-instruct-2501:free", "mistralai/mistral-small-3.1-24b-instruct:free",
    "mistralai/mistral-nemo:free",

    # DeepSeek models
    "deepseek/deepseek-chat", "deepseek/deepseek-chat-v3-0324", "deepseek/deepseek-chat-v3-0324:free", "deepseek/deepseek-chat:free",
    "deepseek/deepseek-r1:free", "deepseek/deepseek-r1-zero:free", "deepseek/deepseek-r1-distill-llama-70b:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free", "deepseek/deepseek-r1-distill-qwen-32b:free", "deepseek/deepseek-v3-base:free",

    # Qwen models
    "qwen/qwen-2.5-72b-instruct:free", "qwen/qwen-2.5-7b-instruct:free", "qwen/qwen-2.5-coder-32b-instruct:free",
    "qwen/qwen-2.5-vl-7b-instruct:free", "qwen/qwen2.5-vl-32b-instruct:free", "qwen/qwen2.5-vl-3b-instruct:free",
    "qwen/qwen2.5-vl-72b-instruct:free", "qwen/qwq-32b:free", "qwen/qwq-32b-preview:free",

    # GLM models
    "thudm/glm-4-32b:free", "thudm/glm-4-9b:free", "thudm/glm-z1-32b", "thudm/glm-z1-32b:free", "thudm/glm-z1-9b:free",

    # Other models
    "microsoft/mai-ds-r1:free", "perplexity/sonar-small-chat", "cohere/command-r",
    "huggingfaceh4/zephyr-7b-beta:free", "microsoft/phi-3-mini-128k-instruct", "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "agentica-org/deepcoder-14b-preview:free", "moonshotai/moonlight-16b-a3b-instruct:free", "shisa-ai/shisa-v2-llama3.3-70b:free",
    "sophosympatheia/rogue-rose-103b-v0.2:free", "tngtech/deepseek-r1t-chimera:free")

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
  } else if (model %in% openrouter_models) {
    return("openrouter")
  }

  # Get list of all supported models
  all_models <- c(
    openai_models, anthropic_models, deepseek_models,
    gemini_models, qwen_models, stepfun_models, zhipu_models, minimax_models, grok_models, openrouter_models
  )

  # Add custom models to the list
  custom_model_names <- ls(envir = custom_models)
  if (length(custom_model_names) > 0) {
    all_models <- c(all_models, custom_model_names)
  }

  # Suggest similar models based on string distance
  suggest_models <- function(input_model, all_models) {
    # Calculate string similarity using edit distance
    similarities <- sapply(all_models, function(m) {
      adist(input_model, m)[1,1] / max(nchar(input_model), nchar(m))
    })

    # Find the most similar models (top 3 or fewer)
    n_suggestions <- min(3, length(all_models))
    if (n_suggestions > 0) {
      most_similar <- all_models[order(similarities)][seq_len(n_suggestions)]
      return(most_similar)
    } else {
      return(character(0))
    }
  }

  # Get model suggestions
  suggestions <- suggest_models(model, all_models)

  # If model not found in any provider's list, show suggestions
  stop("Unsupported model: ", model, "\n",
       "Did you mean one of these? ", paste(suggestions, collapse = ", "), "\n",
       "Or see all supported models: ",
       paste(all_models, collapse = ", "))
}