#' Process request using Kimi models
#'
#' @keywords internal
process_kimi <- function(prompt, model, api_key, base_url = NULL) {
  processor <- KimiProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
