#' Process request using OpenRouter models
#'
#' This function uses the new BaseAPIProcessor architecture for improved
#' maintainability and consistent logging across all API providers.
#'
#' @param prompt Input prompt text
#' @param model Model identifier
#' @param api_key OpenRouter API key
#' @param base_url Optional custom base URL for OpenRouter API
#' @return Processed response as character vector
#' @keywords internal
process_openrouter <- function(prompt, model, api_key, base_url = NULL) {
  # Source the required files with robust path resolution
  script_dir <- if (exists("sys.frame") && !is.null(sys.frame(1)$ofile)) {
    dirname(sys.frame(1)$ofile)
  } else {
    getwd()  # Fallback to current working directory
  }

  if (!exists("BaseAPIProcessor")) {
    source(file.path(script_dir, "base_api_processor.R"))
  }
  if (!exists("OpenRouterProcessor")) {
    source(file.path(script_dir, "openrouter_processor.R"))
  }

  # Create processor and handle request
  # Note: OpenRouterProcessor is defined via source() above
  processor <- get("OpenRouterProcessor")$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}