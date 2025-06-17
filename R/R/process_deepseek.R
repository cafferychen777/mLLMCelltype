#' Process request using DeepSeek models
#'
#' This function uses the new BaseAPIProcessor architecture for improved
#' maintainability and consistent logging across all API providers.
#'
#' @param prompt Input prompt text
#' @param model Model identifier
#' @param api_key DeepSeek API key
#' @return Processed response as character vector
#' @keywords internal
process_deepseek <- function(prompt, model, api_key) {
  # Source the required files with robust path resolution
  script_dir <- if (exists("sys.frame") && !is.null(sys.frame(1)$ofile)) {
    dirname(sys.frame(1)$ofile)
  } else {
    getwd()  # Fallback to current working directory
  }
  
  if (!exists("BaseAPIProcessor")) {
    source(file.path(script_dir, "base_api_processor.R"))
  }
  if (!exists("DeepSeekProcessor")) {
    source(file.path(script_dir, "deepseek_processor.R"))
  }

  # Create processor and handle request
  # Note: DeepSeekProcessor is defined via source() above
  processor <- get("DeepSeekProcessor")$new()
  return(processor$process_request(prompt, model, api_key))
}