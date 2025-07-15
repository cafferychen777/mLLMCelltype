#' Process request using StepFun models
#' 
#' This function uses the new BaseAPIProcessor architecture for improved
#' maintainability and consistent logging across all API providers.
#' 
#' @param prompt Input prompt text
#' @param model Model identifier
#' @param api_key StepFun API key
#' @param base_url Optional custom base URL for StepFun API
#' @return Processed response as character vector
#' @keywords internal
process_stepfun <- function(prompt, model, api_key, base_url = NULL) {
  # Source the required files
  if (!exists("BaseAPIProcessor")) {
    source(file.path(dirname(sys.frame(1)$ofile), "base_api_processor.R"))
  }
  if (!exists("StepFunProcessor")) {
    source(file.path(dirname(sys.frame(1)$ofile), "stepfun_processor.R"))
  }
  
  # Create processor and handle request
  processor <- StepFunProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}