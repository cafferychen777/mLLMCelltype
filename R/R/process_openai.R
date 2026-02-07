#' Process request using OpenAI models
#'
#' @keywords internal
process_openai <- function(prompt, model, api_key, base_url = NULL) {
  processor <- OpenAIProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
