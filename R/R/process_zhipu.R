#' Process request using Zhipu models
#' 
#' This function uses the new BaseAPIProcessor architecture for improved
#' maintainability and consistent logging across all API providers.
#' 
#' @param prompt Input prompt text
#' @param model Model identifier
#' @param api_key Zhipu API key
#' @return Processed response as character vector
#' @keywords internal
process_zhipu <- function(prompt, model, api_key) {
  # Source the required files
  if (!exists("BaseAPIProcessor")) {
    source(file.path(dirname(sys.frame(1)$ofile), "base_api_processor.R"))
  }
  if (!exists("ZhipuProcessor")) {
    source(file.path(dirname(sys.frame(1)$ofile), "zhipu_processor.R"))
  }
  
  # Create processor and handle request
  processor <- ZhipuProcessor$new()
  return(processor$process_request(prompt, model, api_key))
}