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
#'   \item OpenAI: 'chatgpt-4o-latest', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'o1', 'o1-mini', 'o1-pro', 'o3', 'o3-mini', 'o4-mini' and more with date variants
#'   \item Anthropic: 'claude-opus-4-1-20250805', 'claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'
#'   \item DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
#'   \item Google: 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-flash-8b'
#'   \item Qwen: 'qwen-max-2025-01-25', 'qwen3-72b'
#'   \item Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
#'   \item Zhipu: 'glm-4-plus', 'glm-3-turbo'
#'   \item MiniMax: 'minimax-text-01'
#'   \item Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
#'   \item OpenRouter: Provides access to models from multiple providers through a single API. Format: 'provider/model-name'
#'     \itemize{
#'       \item OpenAI models: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#'       \item Anthropic models: 'anthropic/claude-opus-4.1', 'anthropic/claude-opus-4', 'anthropic/claude-sonnet-4', 'anthropic/claude-3.7-sonnet',
#'         'anthropic/claude-3.5-sonnet', 'anthropic/claude-3.5-haiku', 'anthropic/claude-3-opus'
#'       \item Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#'       \item Google models: 'google/gemini-2.5-pro', 'google/gemini-2.5-flash', 'google/gemini-2.0-flash', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#'       \item Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#'       \item Qwen models: 'qwen/qwen3-coder:free', 'qwen/qwen3-235b-a22b-07-25:free', 'qwen/qwen2.5-72b-instruct:free'
#'       \item DeepSeek models: 'deepseek/deepseek-r1:free', 'tngtech/deepseek-r1t2-chimera:free'
#'       \item Other models: 'microsoft/mai-ds-r1:free', 'moonshotai/kimi-k2:free', 'tencent/hunyuan-a13b-instruct:free'
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
  openai_models <- c(
    "chatgpt-4o-latest",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-instruct-0914",
    "gpt-4",
    "gpt-4-0125-preview",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
    "gpt-4o-mini-search-preview",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-search-preview",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-5",
    "gpt-5-2025-08-07",
    "gpt-5-chat-latest",
    "gpt-5-mini",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano",
    "gpt-5-nano-2025-08-07",
    "o1",
    "o1-2024-12-17",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-pro",
    "o1-pro-2025-03-19",
    "o3",
    "o3-2025-04-16",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o4-mini",
    "o4-mini-2025-04-16",
    "o4-mini-deep-research",
    "o4-mini-deep-research-2025-06-26"
  )
  anthropic_models <- c(
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-opus-4",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307"
  )
  deepseek_models <- c("deepseek-chat", "deepseek-reasoner")
  gemini_models <- c(
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b-001"
  )
  qwen_models <- c(
    "qwen-max",
    "qwen-max-2025-01-25",
    "qwen-plus",
    "qwen-turbo",
    "qwen-long",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-14b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-3b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-0.5b-instruct",
    "qwen2-72b-instruct",
    "qwen2-57b-a14b-instruct",
    "qwen2-7b-instruct",
    "qwen2-1.5b-instruct",
    "qwen2-0.5b-instruct",
    "qwen1.5-110b-chat",
    "qwen1.5-72b-chat",
    "qwen1.5-32b-chat",
    "qwen1.5-14b-chat",
    "qwen1.5-7b-chat",
    "qwen1.5-4b-chat",
    "qwen1.5-1.8b-chat",
    "qwen1.5-0.5b-chat",
    "qwen3-72b",
    "qwq-32b-preview"
  )
  stepfun_models <- c("step-2-mini", "step-2-16k", "step-1-8k")
  zhipu_models <- c(
    "glm-4-plus",
    "glm-4",
    "glm-4-0520",
    "glm-4-air",
    "glm-4-airx",
    "glm-4-flash",
    "glm-4-flashx",
    "glm-4v",
    "glm-4v-plus",
    "glm-3-turbo",
    "chatglm3-6b",
    "chatglm2-6b",
    "chatglm-6b",
    "glm-edge"
  )
  minimax_models <- c("minimax-text-01", "minimax-01", "minimax-m1")
  grok_models <- c("grok-3", "grok-3-latest", "grok-3-fast", "grok-3-fast-latest", "grok-3-mini", "grok-3-mini-latest", "grok-3-mini-fast", "grok-3-mini-fast-latest")
  openrouter_models <- c(
    "agentica-org/deepcoder-14b-preview",
    "agentica-org/deepcoder-14b-preview:free",
    "ai21/jamba-large-1.7",
    "ai21/jamba-mini-1.7",
    "aion-labs/aion-1.0",
    "aion-labs/aion-1.0-mini",
    "aion-labs/aion-rp-llama-3.1-8b",
    "alfredpros/codellama-7b-instruct-solidity",
    "alpindale/goliath-120b",
    "amazon/nova-lite-v1",
    "amazon/nova-micro-v1",
    "amazon/nova-pro-v1",
    "anthracite-org/magnum-v2-72b",
    "anthracite-org/magnum-v4-72b",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-opus",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-haiku-20241022",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-sonnet-20240620",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.7-sonnet:thinking",
    "anthropic/claude-opus-4",
    "anthropic/claude-opus-4.1",
    "anthropic/claude-sonnet-4",
    "arcee-ai/coder-large",
    "arcee-ai/maestro-reasoning",
    "arcee-ai/spotlight",
    "arcee-ai/virtuoso-large",
    "arliai/qwq-32b-arliai-rpr-v1",
    "arliai/qwq-32b-arliai-rpr-v1:free",
    "baidu/ernie-4.5-21b-a3b",
    "baidu/ernie-4.5-300b-a47b",
    "baidu/ernie-4.5-vl-28b-a3b",
    "baidu/ernie-4.5-vl-424b-a47b",
    "bytedance/ui-tars-1.5-7b",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "cognitivecomputations/dolphin-mixtral-8x22b",
    "cognitivecomputations/dolphin3.0-mistral-24b",
    "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "cohere/command",
    "cohere/command-a",
    "cohere/command-r",
    "cohere/command-r-03-2024",
    "cohere/command-r-08-2024",
    "cohere/command-r-plus",
    "cohere/command-r-plus-04-2024",
    "cohere/command-r-plus-08-2024",
    "cohere/command-r7b-12-2024",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-prover-v2",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-v3-base",
    "deepseek/deepseek-v3.1-base",
    "eleutherai/llemma_7b",
    "featherless/qwerky-72b:free",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-pro-exp-03-25",
    "google/gemini-2.5-pro-preview",
    "google/gemini-2.5-pro-preview-05-06",
    "google/gemini-flash-1.5",
    "google/gemini-flash-1.5-8b",
    "google/gemini-pro-1.5",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-9b-it:free",
    "google/gemma-3-12b-it",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it",
    "google/gemma-3-4b-it:free",
    "google/gemma-3n-e2b-it:free",
    "google/gemma-3n-e4b-it",
    "google/gemma-3n-e4b-it:free",
    "gryphe/mythomax-l2-13b",
    "inception/mercury",
    "inception/mercury-coder",
    "infermatic/mn-inferor-12b",
    "inflection/inflection-3-pi",
    "inflection/inflection-3-productivity",
    "liquid/lfm-3b",
    "liquid/lfm-7b",
    "mancer/weaver",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3.1-405b",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-405b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.2-11b-vision-instruct",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.2-90b-vision-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-guard-2-8b",
    "meta-llama/llama-guard-3-8b",
    "meta-llama/llama-guard-4-12b",
    "microsoft/mai-ds-r1",
    "microsoft/mai-ds-r1:free",
    "microsoft/phi-3-medium-128k-instruct",
    "microsoft/phi-3-mini-128k-instruct",
    "microsoft/phi-3.5-mini-128k-instruct",
    "microsoft/phi-4",
    "microsoft/phi-4-multimodal-instruct",
    "microsoft/phi-4-reasoning-plus",
    "microsoft/wizardlm-2-8x22b",
    "minimax/minimax-01",
    "minimax/minimax-m1",
    "mistralai/codestral-2501",
    "mistralai/codestral-2508",
    "mistralai/devstral-medium",
    "mistralai/devstral-small",
    "mistralai/devstral-small-2505",
    "mistralai/devstral-small-2505:free",
    "mistralai/magistral-medium-2506",
    "mistralai/magistral-medium-2506:thinking",
    "mistralai/magistral-small-2506",
    "mistralai/ministral-3b",
    "mistralai/ministral-8b",
    "mistralai/mistral-7b-instruct",
    "mistralai/mistral-7b-instruct-v0.1",
    "mistralai/mistral-7b-instruct-v0.3",
    "mistralai/mistral-7b-instruct:free",
    "mistralai/mistral-large",
    "mistralai/mistral-large-2407",
    "mistralai/mistral-large-2411",
    "mistralai/mistral-medium-3",
    "mistralai/mistral-medium-3.1",
    "mistralai/mistral-nemo",
    "mistralai/mistral-nemo:free",
    "mistralai/mistral-saba",
    "mistralai/mistral-small",
    "mistralai/mistral-small-24b-instruct-2501",
    "mistralai/mistral-small-24b-instruct-2501:free",
    "mistralai/mistral-small-3.1-24b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "mistralai/mistral-tiny",
    "mistralai/mixtral-8x22b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/pixtral-12b",
    "mistralai/pixtral-large-2411",
    "moonshotai/kimi-dev-72b:free",
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k2:free",
    "moonshotai/kimi-vl-a3b-thinking",
    "moonshotai/kimi-vl-a3b-thinking:free",
    "morph/morph-v3-fast",
    "morph/morph-v3-large",
    "neversleep/llama-3-lumimaid-70b",
    "neversleep/llama-3.1-lumimaid-8b",
    "neversleep/noromaid-20b",
    "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "nousresearch/deephermes-3-mistral-24b-preview",
    "nousresearch/hermes-2-pro-llama-3-8b",
    "nousresearch/hermes-3-llama-3.1-405b",
    "nousresearch/hermes-3-llama-3.1-70b",
    "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "openai/chatgpt-4o-latest",
    "openai/codex-mini",
    "openai/gpt-3.5-turbo",
    "openai/gpt-3.5-turbo-0613",
    "openai/gpt-3.5-turbo-16k",
    "openai/gpt-3.5-turbo-instruct",
    "openai/gpt-4",
    "openai/gpt-4-0314",
    "openai/gpt-4-1106-preview",
    "openai/gpt-4-turbo",
    "openai/gpt-4-turbo-preview",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-2024-05-13",
    "openai/gpt-4o-2024-08-06",
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-4o-audio-preview",
    "openai/gpt-4o-mini",
    "openai/gpt-4o-mini-2024-07-18",
    "openai/gpt-4o-mini-search-preview",
    "openai/gpt-4o-search-preview",
    "openai/gpt-4o:extended",
    "openai/gpt-5",
    "openai/gpt-5-chat",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-20b:free",
    "openai/o1",
    "openai/o1-mini",
    "openai/o1-mini-2024-09-12",
    "openai/o1-pro",
    "openai/o3",
    "openai/o3-mini",
    "openai/o3-mini-high",
    "openai/o3-pro",
    "openai/o4-mini",
    "openai/o4-mini-high",
    "opengvlab/internvl3-14b",
    "openrouter/auto",
    "perplexity/r1-1776",
    "perplexity/sonar",
    "perplexity/sonar-deep-research",
    "perplexity/sonar-pro",
    "perplexity/sonar-reasoning",
    "perplexity/sonar-reasoning-pro",
    "pygmalionai/mythalion-13b",
    "qwen/qwen-2-72b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-72b-instruct:free",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "qwen/qwen-2.5-vl-7b-instruct",
    "qwen/qwen-max",
    "qwen/qwen-plus",
    "qwen/qwen-turbo",
    "qwen/qwen-vl-max",
    "qwen/qwen-vl-plus",
    "qwen/qwen2.5-vl-32b-instruct",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "qwen/qwen3-14b",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-30b-a3b:free",
    "qwen/qwen3-32b",
    "qwen/qwen3-4b:free",
    "qwen/qwen3-8b",
    "qwen/qwen3-8b:free",
    "qwen/qwen3-coder",
    "qwen/qwen3-coder:free",
    "qwen/qwq-32b",
    "qwen/qwq-32b-preview",
    "qwen/qwq-32b:free",
    "raifle/sorcererlm-8x22b",
    "rekaai/reka-flash-3:free",
    "sao10k/l3-euryale-70b",
    "sao10k/l3-lunaris-8b",
    "sao10k/l3.1-euryale-70b",
    "sao10k/l3.3-euryale-70b",
    "sarvamai/sarvam-m:free",
    "scb10x/llama3.1-typhoon2-70b-instruct",
    "shisa-ai/shisa-v2-llama3.3-70b",
    "shisa-ai/shisa-v2-llama3.3-70b:free",
    "sophosympatheia/midnight-rose-70b",
    "switchpoint/router",
    "tencent/hunyuan-a13b-instruct",
    "tencent/hunyuan-a13b-instruct:free",
    "thedrummer/anubis-70b-v1.1",
    "thedrummer/anubis-pro-105b-v1",
    "thedrummer/rocinante-12b",
    "thedrummer/skyfall-36b-v2",
    "thedrummer/unslopnemo-12b",
    "thudm/glm-4-32b",
    "thudm/glm-4.1v-9b-thinking",
    "thudm/glm-z1-32b",
    "tngtech/deepseek-r1t-chimera",
    "tngtech/deepseek-r1t-chimera:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "undi95/remm-slerp-l2-13b",
    "x-ai/grok-2-1212",
    "x-ai/grok-2-vision-1212",
    "x-ai/grok-3",
    "x-ai/grok-3-beta",
    "x-ai/grok-3-mini",
    "x-ai/grok-3-mini-beta",
    "x-ai/grok-4",
    "x-ai/grok-vision-beta",
    "z-ai/glm-4-32b",
    "z-ai/glm-4.5",
    "z-ai/glm-4.5-air",
    "z-ai/glm-4.5-air:free",
    "z-ai/glm-4.5v")

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