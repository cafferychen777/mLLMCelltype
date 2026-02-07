#' URL Utilities for Base URL Resolution
#'
#' This file contains utility functions for resolving custom base URLs
#' for different API providers.

#' Resolve provider-specific base URL
#'
#' This is the single entry point for all base URL resolution. It resolves
#' the appropriate URL and normalizes it (strips trailing slashes).
#'
#' @param provider Provider name (e.g., "openai", "anthropic")
#' @param base_urls User-provided base URLs: NULL, a single string, or a named list
#' @return Resolved and normalized base URL, or NULL if not specified
#' @keywords internal
resolve_provider_base_url <- function(provider, base_urls) {
  if (is.null(base_urls)) {
    return(NULL)
  }

  url <- NULL

  if (is.character(base_urls) && length(base_urls) == 1) {
    url <- base_urls
  } else if (is.list(base_urls) && provider %in% names(base_urls)) {
    url <- base_urls[[provider]]
  }

  if (is.null(url)) {
    return(NULL)
  }

  # Normalize: strip trailing slashes for consistency
  url <- sub("/+$", "", url)

  return(url)
}
