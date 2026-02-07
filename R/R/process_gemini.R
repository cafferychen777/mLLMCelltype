#' Process request using Gemini models
#'
#' @keywords internal
process_gemini <- function(prompt, model, api_key, base_url = NULL) {
  processor <- GeminiProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
