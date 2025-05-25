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
#'   \item Qwen: 'qwen-max-2025-01-25', 'qwen3-72b'
#'   \item Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
#'   \item Zhipu: 'glm-4-plus', 'glm-3-turbo'
#'   \item MiniMax: 'minimax-text-01'
#'   \item Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
#'   \item OpenRouter: Provides access to models from multiple providers through a single API. Format: 'provider/model-name'
#'     \itemize{
#'       \item OpenAI models: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#'       \item Anthropic models: 'anthropic/claude-3.7-sonnet', 'anthropic/claude-3.5-sonnet',
#'         'anthropic/claude-3.5-haiku', 'anthropic/claude-3-opus'
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
  anthropic_models <- c(
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229"
  )
  deepseek_models <- c("deepseek-chat", "deepseek-reasoner")
  gemini_models <- c("gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash")
  qwen_models <- c("qwen-max-2025-01-25", "qwen3-72b")
  stepfun_models <- c("step-2-mini", "step-2-16k", "step-1-8k")
  zhipu_models <- c("glm-4-plus", "glm-3-turbo")
  minimax_models <- c("minimax-text-01")
  grok_models <- c("grok-3", "grok-3-latest", "grok-3-fast", "grok-3-fast-latest", "grok-3-mini", "grok-3-mini-latest", "grok-3-mini-fast", "grok-3-mini-fast-latest")
  openrouter_models <- c(
    # OpenAI models
    "openai/chatgpt-4o-latest", "openai/codex-mini", "openai/gpt-3.5-turbo",
    "openai/gpt-3.5-turbo-0125", "openai/gpt-3.5-turbo-0613", "openai/gpt-3.5-turbo-1106",
    "openai/gpt-3.5-turbo-16k", "openai/gpt-3.5-turbo-instruct", "openai/gpt-4",
    "openai/gpt-4-0314", "openai/gpt-4-1106-preview", "openai/gpt-4-32k",
    "openai/gpt-4-32k-0314", "openai/gpt-4-turbo", "openai/gpt-4-turbo-preview",
    "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano",
    "openai/gpt-4.5-preview", "openai/gpt-4o", "openai/gpt-4o-2024-05-13",
    "openai/gpt-4o-2024-08-06", "openai/gpt-4o-2024-11-20", "openai/gpt-4o-mini",
    "openai/gpt-4o-mini-2024-07-18", "openai/gpt-4o-mini-search-preview", "openai/gpt-4o-search-preview",
    "openai/gpt-4o:extended", "openai/o1", "openai/o1-mini",
    "openai/o1-mini-2024-09-12", "openai/o1-preview", "openai/o1-preview-2024-09-12",
    "openai/o1-pro", "openai/o3", "openai/o3-mini",
    "openai/o3-mini-high", "openai/o4-mini", "openai/o4-mini-high",

    # Anthropic models
    "anthropic/claude-2", "anthropic/claude-2.0", "anthropic/claude-2.0:beta",
    "anthropic/claude-2.1", "anthropic/claude-2.1:beta", "anthropic/claude-2:beta",
    "anthropic/claude-3-haiku", "anthropic/claude-3-haiku:beta", "anthropic/claude-3-opus",
    "anthropic/claude-3-opus:beta", "anthropic/claude-3-sonnet", "anthropic/claude-3-sonnet:beta",
    "anthropic/claude-3.5-haiku", "anthropic/claude-3.5-haiku-20241022", "anthropic/claude-3.5-haiku-20241022:beta",
    "anthropic/claude-3.5-haiku:beta", "anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-sonnet-20240620",
    "anthropic/claude-3.5-sonnet-20240620:beta", "anthropic/claude-3.5-sonnet:beta", "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.7-sonnet:beta", "anthropic/claude-3.7-sonnet:thinking", "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",

    # Meta-Llama models
    "meta-llama/llama-2-70b-chat", "meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3.1-405b", "meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-3.1-405b:free",
    "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct", "meta-llama/llama-3.2-11b-vision-instruct:free", "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.2-1b-instruct:free", "meta-llama/llama-3.2-3b-instruct", "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.2-90b-vision-instruct", "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.3-8b-instruct:free", "meta-llama/llama-4-maverick", "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout", "meta-llama/llama-4-scout:free", "meta-llama/llama-guard-2-8b",
    "meta-llama/llama-guard-3-8b", "meta-llama/llama-guard-4-12b",

    # Google models
    "google/gemini-2.0-flash-001", "google/gemini-2.0-flash-exp:free", "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.5-flash-preview", "google/gemini-2.5-flash-preview-05-20", "google/gemini-2.5-flash-preview-05-20:thinking",
    "google/gemini-2.5-flash-preview:thinking", "google/gemini-2.5-pro-exp-03-25", "google/gemini-2.5-pro-preview",
    "google/gemini-flash-1.5", "google/gemini-flash-1.5-8b", "google/gemini-pro-1.5",
    "google/gemma-2-27b-it", "google/gemma-2-9b-it", "google/gemma-2-9b-it:free",
    "google/gemma-3-12b-it", "google/gemma-3-12b-it:free", "google/gemma-3-1b-it:free",
    "google/gemma-3-27b-it", "google/gemma-3-27b-it:free", "google/gemma-3-4b-it",
    "google/gemma-3-4b-it:free", "google/gemma-3n-e4b-it:free",

    # Mistral models
    "mistralai/codestral-2501", "mistralai/codestral-mamba", "mistralai/devstral-small",
    "mistralai/devstral-small:free", "mistralai/ministral-3b", "mistralai/ministral-8b",
    "mistralai/mistral-7b-instruct", "mistralai/mistral-7b-instruct-v0.1", "mistralai/mistral-7b-instruct-v0.2",
    "mistralai/mistral-7b-instruct-v0.3", "mistralai/mistral-7b-instruct:free", "mistralai/mistral-large",
    "mistralai/mistral-large-2407", "mistralai/mistral-large-2411", "mistralai/mistral-medium",
    "mistralai/mistral-medium-3", "mistralai/mistral-nemo", "mistralai/mistral-nemo:free",
    "mistralai/mistral-saba", "mistralai/mistral-small", "mistralai/mistral-small-24b-instruct-2501",
    "mistralai/mistral-small-24b-instruct-2501:free", "mistralai/mistral-small-3.1-24b-instruct", "mistralai/mistral-small-3.1-24b-instruct:free",
    "mistralai/mistral-tiny", "mistralai/mixtral-8x22b-instruct", "mistralai/mixtral-8x7b-instruct",
    "mistralai/pixtral-12b", "mistralai/pixtral-large-2411",

    # DeepSeek models
    "deepseek/deepseek-chat", "deepseek/deepseek-chat-v3-0324", "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-chat:free", "deepseek/deepseek-coder", "deepseek/deepseek-prover-v2",
    "deepseek/deepseek-prover-v2:free", "deepseek/deepseek-r1", "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1-distill-llama-70b:free", "deepseek/deepseek-r1-distill-llama-8b", "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek/deepseek-r1-distill-qwen-14b", "deepseek/deepseek-r1-distill-qwen-14b:free", "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek/deepseek-r1-distill-qwen-32b:free", "deepseek/deepseek-r1-zero:free", "deepseek/deepseek-r1:free",
    "deepseek/deepseek-v3-base:free",

    # Qwen models
    "qwen/qwen-2-72b-instruct", "qwen/qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct:free",
    "qwen/qwen-2.5-7b-instruct", "qwen/qwen-2.5-7b-instruct:free", "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct:free", "qwen/qwen-2.5-vl-7b-instruct", "qwen/qwen-2.5-vl-7b-instruct:free",
    "qwen/qwen-max", "qwen/qwen-plus", "qwen/qwen-turbo",
    "qwen/qwen-vl-max", "qwen/qwen-vl-plus", "qwen/qwen2.5-coder-7b-instruct",
    "qwen/qwen2.5-vl-32b-instruct", "qwen/qwen2.5-vl-32b-instruct:free", "qwen/qwen2.5-vl-3b-instruct:free",
    "qwen/qwen2.5-vl-72b-instruct", "qwen/qwen2.5-vl-72b-instruct:free", "qwen/qwen3-14b",
    "qwen/qwen3-14b:free", "qwen/qwen3-235b-a22b", "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-30b-a3b", "qwen/qwen3-30b-a3b:free", "qwen/qwen3-32b",
    "qwen/qwen3-32b:free", "qwen/qwen3-8b", "qwen/qwen3-8b:free",
    "qwen/qwq-32b", "qwen/qwq-32b-preview", "qwen/qwq-32b:free",

    # GLM models
    "thudm/glm-4-32b", "thudm/glm-4-32b:free", "thudm/glm-z1-32b",
    "thudm/glm-z1-32b:free", "thudm/glm-z1-rumination-32b",

    # Microsoft models
    "microsoft/mai-ds-r1:free", "microsoft/phi-3-medium-128k-instruct", "microsoft/phi-3-mini-128k-instruct",
    "microsoft/phi-3.5-mini-128k-instruct", "microsoft/phi-4", "microsoft/phi-4-multimodal-instruct",
    "microsoft/phi-4-reasoning-plus", "microsoft/phi-4-reasoning-plus:free", "microsoft/phi-4-reasoning:free",
    "microsoft/wizardlm-2-8x22b",

    # Other models
    "01-ai/yi-large", "aetherwiing/mn-starcannon-12b", "agentica-org/deepcoder-14b-preview:free",
    "ai21/jamba-1.6-large", "ai21/jamba-1.6-mini", "aion-labs/aion-1.0",
    "aion-labs/aion-1.0-mini", "aion-labs/aion-rp-llama-3.1-8b", "alfredpros/codellama-7b-instruct-solidity",
    "all-hands/openhands-lm-32b-v0.1", "allenai/olmo-7b-instruct", "alpindale/goliath-120b",
    "alpindale/magnum-72b", "amazon/nova-lite-v1", "amazon/nova-micro-v1",
    "amazon/nova-pro-v1", "anthracite-org/magnum-v2-72b", "anthracite-org/magnum-v4-72b",
    "arcee-ai/arcee-blitz", "arcee-ai/caller-large", "arcee-ai/coder-large",
    "arcee-ai/maestro-reasoning", "arcee-ai/spotlight", "arcee-ai/virtuoso-large",
    "arcee-ai/virtuoso-medium-v2", "arliai/qwq-32b-arliai-rpr-v1:free", "cognitivecomputations/dolphin-mixtral-8x22b",
    "cognitivecomputations/dolphin3.0-mistral-24b:free", "cognitivecomputations/dolphin3.0-r1-mistral-24b:free", "cohere/command",
    "cohere/command-a", "cohere/command-r", "cohere/command-r-03-2024",
    "cohere/command-r-08-2024", "cohere/command-r-plus", "cohere/command-r-plus-04-2024",
    "cohere/command-r-plus-08-2024", "cohere/command-r7b-12-2024", "eleutherai/llemma_7b",
    "eva-unit-01/eva-llama-3.33-70b", "eva-unit-01/eva-qwen-2.5-32b", "eva-unit-01/eva-qwen-2.5-72b",
    "featherless/qwerky-72b:free", "gryphe/mythomax-l2-13b", "inception/mercury-coder-small-beta",
    "infermatic/mn-inferor-12b", "inflection/inflection-3-pi", "inflection/inflection-3-productivity",
    "liquid/lfm-3b", "liquid/lfm-40b", "liquid/lfm-7b",
    "mancer/weaver", "minimax/minimax-01", "moonshotai/kimi-vl-a3b-thinking:free",
    "moonshotai/moonlight-16b-a3b-instruct:free", "neversleep/llama-3-lumimaid-70b", "neversleep/llama-3-lumimaid-8b",
    "neversleep/llama-3-lumimaid-8b:extended", "neversleep/llama-3.1-lumimaid-70b", "neversleep/llama-3.1-lumimaid-8b",
    "neversleep/noromaid-20b", "nothingiisreal/mn-celeste-12b", "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "nousresearch/deephermes-3-mistral-24b-preview:free", "nousresearch/hermes-2-pro-llama-3-8b", "nousresearch/hermes-3-llama-3.1-405b",
    "nousresearch/hermes-3-llama-3.1-70b", "nousresearch/nous-hermes-2-mixtral-8x7b-dpo", "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free", "nvidia/llama-3.3-nemotron-super-49b-v1", "nvidia/llama-3.3-nemotron-super-49b-v1:free",
    "open-r1/olympiccoder-32b:free", "opengvlab/internvl3-14b:free", "opengvlab/internvl3-2b:free",
    "openrouter/auto", "perplexity/llama-3.1-sonar-large-128k-online", "perplexity/llama-3.1-sonar-small-128k-online",
    "perplexity/r1-1776", "perplexity/sonar", "perplexity/sonar-deep-research",
    "perplexity/sonar-pro", "perplexity/sonar-reasoning", "perplexity/sonar-reasoning-pro",
    "pygmalionai/mythalion-13b", "raifle/sorcererlm-8x22b", "rekaai/reka-flash-3:free",
    "sao10k/fimbulvetr-11b-v2", "sao10k/l3-euryale-70b", "sao10k/l3-lunaris-8b",
    "sao10k/l3.1-euryale-70b", "sao10k/l3.3-euryale-70b", "sarvamai/sarvam-m",
    "scb10x/llama3.1-typhoon2-70b-instruct", "scb10x/llama3.1-typhoon2-8b-instruct", "shisa-ai/shisa-v2-llama3.3-70b:free",
    "sophosympatheia/midnight-rose-70b", "thedrummer/anubis-pro-105b-v1", "thedrummer/rocinante-12b",
    "thedrummer/skyfall-36b-v2", "thedrummer/unslopnemo-12b", "thedrummer/valkyrie-49b-v1",
    "tngtech/deepseek-r1t-chimera:free", "undi95/remm-slerp-l2-13b", "undi95/toppy-m-7b",
    "x-ai/grok-2-1212", "x-ai/grok-2-vision-1212", "x-ai/grok-3-beta",
    "x-ai/grok-3-mini-beta", "x-ai/grok-beta", "x-ai/grok-vision-beta")

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