
#' Custom model manager for mLLMCelltype
#'
#' This module provides functionality to register and manage custom LLM
#' providers
#' and models. It allows users to integrate their own LLM services with the
#' mLLMCelltype framework.
#'
#' @keywords internal

#' Environment to store custom providers and their configurations
custom_providers <- new.env(parent = emptyenv())
custom_models <- new.env(parent = emptyenv())

#' Register a custom LLM provider
#'
#' @param provider_name Unique name for the custom provider
#' @param process_fn Function that processes LLM requests. Must accept parameters: prompt, model, api_key
#' @param description Optional description of the provider
#'
#' @return Invisible NULL
#' @export
#'
#' @examples
#' \dontrun{
#' register_custom_provider(
#'   provider_name = "my_provider",
#'   process_fn = function(prompt, model, api_key) {
#'     # Custom implementation
#'     response <- httr::POST(
#'       url = "your_api_endpoint",
#'       body = list(prompt = prompt),
#'       encode = "json"
#'     )
#'     return(httr::content(response)$choices[[1]]$text)
#'   }
#' )
#' }
register_custom_provider <- function(provider_name, process_fn,
                                     description = NULL) {
  # Input validation
  if (!is.character(provider_name) || length(provider_name) != 1) {
    stop("provider_name must be a single character string")
  }
  if (!is.function(process_fn)) {
    stop("process_fn must be a function")
  }

  # Normalize provider name to lowercase for consistent lookup
  provider_name <- tolower(provider_name)

  # Check if provider already exists
  if (exists(provider_name, envir = custom_providers)) {
    stop("Provider '", provider_name, "' already exists")
  }

  # Validate process_fn arguments
  fn_args <- names(formals(process_fn))
  required_args <- c("prompt", "model", "api_key")
  if (!all(required_args %in% fn_args)) {
    stop("process_fn must accept parameters: ",
         paste(required_args, collapse = ", "))
  }

  # Store provider configuration
  assign(provider_name,
         list(
           process_fn = process_fn,
           description = description,
           models = character(0)
         ),
         envir = custom_providers)

  get_logger()$info("Registered custom provider", list(provider = provider_name))
  invisible(TRUE)
}

#' Register a custom model for a provider
#'
#' @param model_name Unique name for the custom model
#' @param provider_name Name of the provider this model belongs to
#' @param model_config List of configuration parameters for the model (e.g., temperature, max_tokens)
#'
#' @return Invisible TRUE on success
#' @export
#'
#' @examples
#' \dontrun{
#' register_custom_model(
#'   model_name = "my_model",
#'   provider_name = "my_provider",
#'   model_config = list(
#'     temperature = 0.7,
#'     max_tokens = 2000
#'   )
#' )
#' }
register_custom_model <- function(model_name, provider_name,
                                  model_config = list()) {
  # Input validation
  if (!is.character(model_name) || length(model_name) != 1) {
    stop("model_name must be a single character string")
  }
  if (!is.character(provider_name) || length(provider_name) != 1) {
    stop("provider_name must be a single character string")
  }
  if (!is.list(model_config)) {
    stop("model_config must be a list")
  }

  # Normalize names to lowercase for consistent lookup with get_provider()
  model_name <- tolower(model_name)
  provider_name <- tolower(provider_name)

  # Check if provider exists
  if (!exists(provider_name, envir = custom_providers)) {
    stop("Provider '", provider_name, "' does not exist")
  }

  # Check if model already exists
  if (exists(model_name, envir = custom_models)) {
    stop("Model '", model_name, "' already exists")
  }

  # Store model configuration
  assign(model_name,
         list(
           provider = provider_name,
           config = model_config
         ),
         envir = custom_models)

  # Update provider's model list
  provider_data <- get(provider_name, envir = custom_providers)
  provider_data$models <- c(provider_data$models, model_name)
  assign(provider_name, provider_data, envir = custom_providers)

  get_logger()$info("Registered custom model for provider",
           list(model = model_name, provider = provider_name))
  invisible(TRUE)
}

#' Process request using custom provider
#' @keywords internal
process_custom <- function(prompt, model, api_key) {
  # Normalize to lowercase for consistent lookup with get_provider()
  model_lower <- tolower(model)

  # Check if model exists
  if (!exists(model_lower, envir = custom_models)) {
    stop("Model '", model, "' not found")
  }

  # Get model and provider data
  model_data <- get(model_lower, envir = custom_models)
  provider_data <- get(model_data$provider, envir = custom_providers)

  # Call provider's process function
  get_logger()$info("Processing request with custom model", list(model = model))
  tryCatch({
    response <- provider_data$process_fn(prompt, model, api_key)
    get_logger()$info("Custom model request processed successfully")
    return(response)
  }, error = function(e) {
    get_logger()$error("Error processing custom model request",
              list(error = e$message))
    stop("Failed to process request with custom model: ", e$message)
  })
}

#' Get list of registered custom providers
#
#' @export
list_custom_providers <- function() {
  ls(envir = custom_providers)
}

#' Get list of registered custom models
#
#' @export
list_custom_models <- function() {
  ls(envir = custom_models)
}
