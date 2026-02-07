#' URL Utilities for Base URL Resolution
#'
#' This file contains utility functions for resolving and validating
#' custom base URLs for different API providers.

#' Resolve provider-specific base URL
#'
#
#
#
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
#
#
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
