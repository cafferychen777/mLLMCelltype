#' Utility functions for API key management
#' 
#' This file contains utility functions for managing API keys and related operations.

#' Get API key for a specific model
#' 
#' This function retrieves the appropriate API key for a given model by first checking
#' the provider name and then the model name in the provided API keys list.
#' 
#' @param model Model name to get API key for
#' @param api_keys Named list of API keys with provider or model names as keys
#'
#' @return API key string for the specified model
#' @export
get_api_key <- function(model, api_keys) {
  provider <- get_provider(model)
  is_valid_key <- function(key) {
    is.character(key) && length(key) == 1 && !is.na(key) && nzchar(trimws(key))
  }

  # First try to get by provider name
  if (provider %in% names(api_keys)) {
    key <- api_keys[[provider]]
    if (is_valid_key(key)) return(trimws(key))
  }

  # If not found, try to get by model name
  if (model %in% names(api_keys)) {
    key <- api_keys[[model]]
    if (is_valid_key(key)) return(trimws(key))
  }

  # If still not found or all keys empty, return NULL
  return(NULL)
}
