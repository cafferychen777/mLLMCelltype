#' Process request using DeepSeek models
#'
#' @keywords internal
process_deepseek <- function(prompt, model, api_key, base_url = NULL) {
  processor <- DeepSeekProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
