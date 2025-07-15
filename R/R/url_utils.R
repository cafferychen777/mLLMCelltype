#' URL Utilities for Base URL Resolution
#' 
#' This file contains utility functions for resolving and validating
#' custom base URLs for different API providers.

#' Resolve provider-specific base URL
#' 
#' @param provider Provider name (e.g., "openai", "anthropic")
#' @param base_urls User-provided base URLs (string or named list)
#' @return Resolved base URL or NULL
#' @keywords internal
resolve_provider_base_url <- function(provider, base_urls) {
  if (is.null(base_urls)) {
    return(NULL)
  }
  
  if (is.character(base_urls) && length(base_urls) == 1) {
    # Single URL for all providers
    return(base_urls)
  }
  
  if (is.list(base_urls) && provider %in% names(base_urls)) {
    # Provider-specific URL
    return(base_urls[[provider]])
  }
  
  return(NULL)
}

#' Validate base URL format
#' 
#' @param url URL to validate
#' @return TRUE if valid, FALSE otherwise
#' @keywords internal
validate_base_url <- function(url) {
  if (is.null(url) || !is.character(url) || length(url) != 1) {
    return(FALSE)
  }
  
  # Basic URL validation - must start with http:// or https://
  if (!grepl("^https?://", url)) {
    return(FALSE)
  }
  
  # Additional validation - should not end with slash for consistency
  if (grepl("/$", url)) {
    warning("Base URL should not end with '/'. Removing trailing slash.")
    return(TRUE)
  }
  
  return(TRUE)
}

#' Sanitize base URL
#' 
#' @param url URL to sanitize
#' @return Sanitized URL
#' @keywords internal
sanitize_base_url <- function(url) {
  if (is.null(url)) {
    return(NULL)
  }
  
  # Remove trailing slash if present
  url <- gsub("/$", "", url)
  
  return(url)
}

#' Get provider from model name
#' 
#' This is a helper function that extracts the provider name from a model identifier.
#' It's used internally to determine which base_url to use from a list of provider-specific URLs.
#' 
#' @param model Model identifier
#' @return Provider name
#' @keywords internal
get_provider <- function(model) {
  # OpenAI models
  if (grepl("^(gpt-|o1-|text-|davinci|curie|babbage|ada)", model)) {
    return("openai")
  }
  
  # Anthropic models
  if (grepl("^claude-", model)) {
    return("anthropic")
  }
  
  # DeepSeek models
  if (grepl("^deepseek-", model)) {
    return("deepseek")
  }
  
  # Gemini models
  if (grepl("^(gemini-|models/gemini)", model)) {
    return("gemini")
  }
  
  # Grok models
  if (grepl("^grok-", model)) {
    return("grok")
  }
  
  # OpenRouter models (contain slash)
  if (grepl("/", model)) {
    return("openrouter")
  }
  
  # Qwen models
  # Note: Qwen has two API endpoints:
  # - Domestic (China): https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
  # - International: https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation
  if (grepl("^qwen", model, ignore.case = TRUE)) {
    return("qwen")
  }
  
  # StepFun models
  if (grepl("^step-", model)) {
    return("stepfun")
  }
  
  # Zhipu models
  if (grepl("^(glm-|chatglm)", model)) {
    return("zhipu")
  }
  
  # Minimax models
  if (grepl("^(abab|minimax)", model)) {
    return("minimax")
  }
  
  # Default fallback
  warning(paste("Unknown model provider for model:", model, ". Defaulting to 'openai'."))
  return("openai")
}
