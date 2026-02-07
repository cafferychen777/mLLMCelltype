#' Process request using StepFun models
#'
#' @keywords internal
process_stepfun <- function(prompt, model, api_key, base_url = NULL) {
  processor <- StepFunProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
