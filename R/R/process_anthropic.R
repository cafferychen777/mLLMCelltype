#' Process request using Anthropic models
#'
#' @keywords internal
process_anthropic <- function(prompt, model, api_key, base_url = NULL) {
  processor <- AnthropicProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
