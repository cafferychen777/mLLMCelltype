#' Process request using Gemini models
#' 
#' This function uses the new BaseAPIProcessor architecture for improved
#' maintainability and consistent logging across all API providers.
#' 
#' @param prompt Input prompt text
#' @param model Model identifier
#' @param api_key Gemini API key
#' @param base_url Optional custom base URL for Gemini API
#' @return Processed response as character vector
#' @keywords internal
process_gemini <- function(prompt, model, api_key, base_url = NULL) {
  # Source the required files
  if (!exists("BaseAPIProcessor")) {
    source(file.path(dirname(sys.frame(1)$ofile), "base_api_processor.R"))
  }
  if (!exists("GeminiProcessor")) {
    source(file.path(dirname(sys.frame(1)$ofile), "gemini_processor.R"))
  }
  
  # Create processor and handle request
  processor <- GeminiProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}