#' Process request using Grok models
#' 
#' This function uses the new BaseAPIProcessor architecture for improved
#' maintainability and consistent logging across all API providers.
#' 
#' @param prompt Input prompt text
#' @param model Model identifier
#' @param api_key Grok API key
#' @return Processed response as character vector
#' @keywords internal
process_grok <- function(prompt, model, api_key) {
  # Source the required files with robust path resolution
  script_dir <- if (exists("sys.frame") && !is.null(sys.frame(1)$ofile)) {
    dirname(sys.frame(1)$ofile)
  } else {
    getwd()  # Fallback to current working directory
  }
  
  if (!exists("BaseAPIProcessor")) {
    source(file.path(script_dir, "base_api_processor.R"))
  }
  if (!exists("GrokProcessor")) {
    source(file.path(script_dir, "grok_processor.R"))
  }
  
  # Create processor and handle request
  # Note: GrokProcessor is defined via source() above
  processor <- get("GrokProcessor")$new()
  return(processor$process_request(prompt, model, api_key))
}