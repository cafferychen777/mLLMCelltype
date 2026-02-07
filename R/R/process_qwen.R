#' Process request using Qwen models
#'
#' @keywords internal
process_qwen <- function(prompt, model, api_key, base_url = NULL) {
  processor <- QwenProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
