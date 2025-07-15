#' Base API Processor Class
#' 
#' Abstract base class for API processors that provides common functionality
#' including unified logging, error handling, input processing, and response validation.
#' This eliminates code duplication across all provider-specific processors.
#'
#' @importFrom R6 R6Class
#' @export
BaseAPIProcessor <- R6::R6Class("BaseAPIProcessor",
  public = list(
    #' @field provider_name Name of the API provider
    provider_name = NULL,

    #' @field logger Unified logger instance
    logger = NULL,

    #' @field base_url Custom base URL for API endpoints
    base_url = NULL,

    #' @description
    #' Initialize the base API processor
    #' @param provider_name Name of the API provider (e.g., "openai", "anthropic")
    #' @param base_url Optional custom base URL for API endpoints
    initialize = function(provider_name, base_url = NULL) {
      self$provider_name <- provider_name
      self$base_url <- base_url
      self$logger <- get_logger()
      self$logger$info(sprintf("Initialized %s processor", provider_name),
                      list(provider = provider_name, custom_url = !is.null(base_url)))
    },
    
    #' @description
    #' Main entry point for processing API requests
    #' @param prompt Input prompt text
    #' @param model Model identifier
    #' @param api_key API key for authentication
    #' @return Processed response as character vector
    process_request = function(prompt, model, api_key) {
      start_time <- Sys.time()
      
      self$logger$info(sprintf("Starting %s API request", self$provider_name), 
                      list(model = model, provider = self$provider_name))
      
      tryCatch({
        # Validate inputs
        private$validate_inputs(prompt, model, api_key)
        
        # Process input into chunks
        input_chunks <- private$process_input(prompt)
        
        # Process all chunks
        all_results <- private$process_chunks(input_chunks, model, api_key)
        
        # Validate and consolidate results
        final_result <- private$consolidate_results(all_results)
        
        # Log success
        duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        self$logger$log_api_call(self$provider_name, model, duration, TRUE)
        
        return(final_result)
        
      }, error = function(e) {
        duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        self$logger$log_api_call(self$provider_name, model, duration, FALSE)
        self$logger$error(sprintf("%s API request failed: %s", self$provider_name, e$message),
                         list(provider = self$provider_name, model = model, error = e$message))
        stop(sprintf("%s API request failed: %s", self$provider_name, e$message))
      })
    },

    #' @description
    #' Get the API URL to use for requests
    #' @return API URL string
    get_api_url = function() {
      if (!is.null(self$base_url)) {
        self$logger$debug("Using custom base URL",
                         list(provider = self$provider_name, url = self$base_url))
        return(self$base_url)
      }
      return(self$get_default_api_url())
    },

    #' @description
    #' Abstract method to be implemented by subclasses for getting default API URL
    #' @return Default API URL string
    get_default_api_url = function() {
      stop("get_default_api_url must be implemented by subclass")
    },

    #' @description
    #' Abstract method to be implemented by subclasses for making the actual API call
    #' @param chunk_content Content for this chunk
    #' @param model Model identifier
    #' @param api_key API key
    #' @return Raw API response
    make_api_call = function(chunk_content, model, api_key) {
      stop("make_api_call must be implemented by subclass")
    },
    
    #' @description
    #' Abstract method to be implemented by subclasses for extracting content from response
    #' @param response Raw API response
    #' @param model Model identifier
    #' @return Extracted text content
    extract_response_content = function(response, model) {
      stop("extract_response_content must be implemented by subclass")
    }
  ),
  
  private = list(
    #' Validate input parameters
    #' @param prompt Input prompt
    #' @param model Model identifier
    #' @param api_key API key
    validate_inputs = function(prompt, model, api_key) {
      if (is.null(api_key) || api_key == "") {
        self$logger$error(sprintf("%s API key is missing or empty", self$provider_name),
                         list(provider = self$provider_name))
        stop(sprintf("%s API key is required but not provided", self$provider_name))
      }
      
      if (is.null(prompt) || prompt == "") {
        self$logger$error("Prompt is missing or empty",
                         list(provider = self$provider_name))
        stop("Prompt is required but not provided")
      }
      
      if (is.null(model) || model == "") {
        self$logger$error("Model is missing or empty",
                         list(provider = self$provider_name))
        stop("Model is required but not provided")
      }
      
      self$logger$debug("Input validation passed",
                       list(provider = self$provider_name, model = model))
    },
    
    #' Process input text into chunks
    #' @param prompt Input prompt text
    #' @return List with input_lines and chunk_ids
    process_input = function(prompt) {
      input_lines <- strsplit(prompt, "\n")[[1]]
      cutnum <- 1  # Always use 1 chunk for consistency
      
      self$logger$debug("Processing input into chunks",
                       list(provider = self$provider_name, 
                            lines_count = length(input_lines),
                            chunk_count = cutnum))
      
      if (cutnum > 1) {
        cid <- as.numeric(cut(1:length(input_lines), cutnum))
      } else {
        cid <- rep(1, length(input_lines))
      }
      
      return(list(
        input_lines = input_lines,
        chunk_ids = cid,
        chunk_count = cutnum
      ))
    },
    
    #' Process all input chunks
    #' @param input_chunks Processed input chunks
    #' @param model Model identifier
    #' @param api_key API key
    #' @return List of results from all chunks
    process_chunks = function(input_chunks, model, api_key) {
      all_results <- sapply(1:input_chunks$chunk_count, function(i) {
        self$logger$debug("Processing chunk",
                         list(current_chunk = i, 
                              total_chunks = input_chunks$chunk_count,
                              provider = self$provider_name))
        
        # Get lines for this chunk
        chunk_line_ids <- which(input_chunks$chunk_ids == i)
        chunk_content <- paste(input_chunks$input_lines[chunk_line_ids], collapse = '\n')
        
        tryCatch({
          # Make API call (implemented by subclass)
          response <- self$make_api_call(chunk_content, model, api_key)
          
          # Extract content (implemented by subclass)
          content <- self$extract_response_content(response, model)
          
          # Log complete API request and response for audit/debugging
          api_call_id <- self$logger$log_api_request_response(
            provider = self$provider_name,
            model = model,
            prompt_content = chunk_content,
            response_content = content,
            request_metadata = list(
              chunk_number = i,
              total_chunks = input_chunks$chunk_count,
              chunk_line_ids = chunk_line_ids
            ),
            response_metadata = list(
              raw_response_class = class(response),
              extracted_content_length = if(is.character(content)) length(content) else 1
            )
          )
          
          # Process the content
          private$process_response_content(content, model)
          
        }, error = function(e) {
          self$logger$error(sprintf("Failed to process chunk %d: %s", i, e$message),
                           list(provider = self$provider_name, 
                                model = model,
                                chunk = i,
                                error = e$message))
          return(NULL)
        })
      }, simplify = FALSE)
      
      return(all_results)
    },
    
    #' Process response content into lines
    #' @param response_content Raw response content
    #' @param model Model identifier
    #' @return Processed response lines
    process_response_content = function(response_content, model) {
      if (!is.character(response_content)) {
        self$logger$error("Response content is not a character string",
                         list(provider = self$provider_name,
                              model = model,
                              response_type = typeof(response_content)))
        return(c("Error: Invalid response format"))
      }
      
      tryCatch({
        res <- strsplit(response_content, '\n')[[1]]
        self$logger$debug(sprintf("Processed response from %s", self$provider_name),
                         list(provider = self$provider_name,
                              model = model,
                              lines_count = length(res),
                              response_length = nchar(response_content)))
        return(res)
      }, error = function(e) {
        self$logger$error("Failed to split response content",
                         list(provider = self$provider_name,
                              model = model,
                              error = e$message))
        return(c("Error: Failed to parse response"))
      })
    },
    
    #' Consolidate results from all chunks
    #' @param all_results List of results from all chunks
    #' @return Final consolidated result
    consolidate_results = function(all_results) {
      self$logger$info("All chunks processed, consolidating results",
                      list(provider = self$provider_name,
                           chunks_processed = length(all_results)))
      
      # Filter out NULL values
      valid_results <- all_results[!sapply(all_results, is.null)]
      
      if (length(valid_results) == 0) {
        self$logger$error("No valid responses received",
                         list(provider = self$provider_name,
                              chunks_attempted = length(all_results)))
        return(c("Error: No valid responses"))
      }
      
      # Clean up and return results
      final_result <- gsub(',$', '', unlist(valid_results))
      
      self$logger$info("Results consolidated successfully",
                      list(provider = self$provider_name,
                           valid_chunks = length(valid_results),
                           total_lines = length(final_result)))
      
      return(final_result)
    }
  )
)