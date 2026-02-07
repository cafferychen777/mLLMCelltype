#' Process request using Grok models
#'
#' @keywords internal
process_grok <- function(prompt, model, api_key, base_url = NULL) {
  processor <- GrokProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
