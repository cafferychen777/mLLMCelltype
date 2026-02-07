#' Process request using OpenRouter models
#'
#' @keywords internal
process_openrouter <- function(prompt, model, api_key, base_url = NULL) {
  processor <- OpenRouterProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
