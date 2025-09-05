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
  
  # First try to get by provider name
  if (provider %in% names(api_keys)) {
    return(api_keys[[provider]])
  }
  
  # If not found, try to get by model name
  if (model %in% names(api_keys)) {
    return(api_keys[[model]])
  }
  
  # If still not found, return NULL
  return(NULL)
}
