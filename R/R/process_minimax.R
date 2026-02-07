#' Process request using MiniMax models
#'
#' @keywords internal
process_minimax <- function(prompt, model, api_key, base_url = NULL) {
  processor <- MinimaxProcessor$new(base_url = base_url)
  return(processor$process_request(prompt, model, api_key))
}
