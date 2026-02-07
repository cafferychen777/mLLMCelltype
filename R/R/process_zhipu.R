#' Process request using Zhipu models
#'
#' @keywords internal
process_zhipu <- function(prompt, model, api_key, base_url = NULL) {
  processor <- ZhipuProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
