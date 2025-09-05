#' Unified Logger for mLLMCelltype Package
#' 
#' This logger provides centralized, multi-level logging with structured output,
#' log rotation, and performance monitoring capabilities.
#'
#' @importFrom R6 R6Class
#' @importFrom jsonlite toJSON
#' @export
UnifiedLogger <- R6::R6Class("UnifiedLogger",
  public = list(
    #' @field log_dir Directory for storing log files
    log_dir = NULL,
    
    #' @field log_level Current logging level
    log_level = NULL,
    
    #' @field session_id Unique identifier for the current session
    session_id = NULL,
    
    #' @field max_log_size Maximum log file size in MB (default: 10MB)
    max_log_size = 10,
    
    #' @field max_log_files Maximum number of log files to keep (default: 5)
    max_log_files = 5,
    
    #' @field enable_console Whether to output to console (default: TRUE)
    enable_console = TRUE,
    
    #' @field enable_json Whether to use JSON format (default: TRUE)
    enable_json = TRUE,
    
    #' @field performance_stats Performance monitoring statistics
    performance_stats = NULL,
    
    #' @description
    #' Initialize the unified logger
    #
    #
    #
    #
    #
    #
    initialize = function(base_dir = "logs", level = "INFO", max_size = 10, 
                         max_files = 5, console_output = TRUE, json_format = TRUE) {
      self$log_dir <- base_dir
      self$log_level <- toupper(level)
      self$max_log_size <- max_size
      self$max_log_files <- max_files
      self$enable_console <- console_output
      self$enable_json <- json_format
      
      # Only create log directory if we're not in R CMD check environment
      # This prevents the logs directory from being created during package checks
      if (!nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_", "")) && !dir.exists(self$log_dir)) {
        dir.create(self$log_dir, recursive = TRUE)
      }
      
      # Generate session ID
      self$session_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
      
      # Initialize performance stats
      self$performance_stats <- list(
        session_start = Sys.time(),
        api_calls = 0,
        cache_hits = 0,
        cache_misses = 0,
        errors = 0,
        warnings = 0
      )
      
      # Log initialization
      self$info("Unified logger initialized", list(
        session_id = self$session_id,
        log_level = self$log_level,
        log_dir = self$log_dir
      ))
    },
    
    #' @description
    #' Log a debug message
    #
    #
    debug = function(message, context = NULL) {
      private$log_message("DEBUG", message, context)
    },
    
    #' @description
    #' Log an info message
    #
    #
    info = function(message, context = NULL) {
      private$log_message("INFO", message, context)
    },
    
    #' @description
    #' Log a warning message
    #
    #
    warn = function(message, context = NULL) {
      self$performance_stats$warnings <- self$performance_stats$warnings + 1
      private$log_message("WARN", message, context)
    },
    
    #' @description
    #' Log an error message
    #
    #
    error = function(message, context = NULL) {
      self$performance_stats$errors <- self$performance_stats$errors + 1
      private$log_message("ERROR", message, context)
    },
    
    #' @description
    #' Log API call performance
    #
    #
    #
    #
    #
    log_api_call = function(provider, model, duration, success = TRUE, tokens = NULL) {
      self$performance_stats$api_calls <- self$performance_stats$api_calls + 1
      
      context <- list(
        provider = provider,
        model = model,
        duration_seconds = round(duration, 3),
        success = success,
        tokens = tokens,
        call_count = self$performance_stats$api_calls
      )
      
      if (success) {
        self$info(sprintf("API call completed: %s/%s (%.3fs)", provider, model, duration), context)
      } else {
        self$error(sprintf("API call failed: %s/%s (%.3fs)", provider, model, duration), context)
      }
    },
    
    #' @description
    #' Log complete API request and response for debugging and audit
    #
    #
    #
    #
    #
    #
    log_api_request_response = function(provider, model, prompt_content, response_content, 
                                      request_metadata = NULL, response_metadata = NULL) {
      # Skip during R CMD check
      if (nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_", ""))) {
        return(invisible(NULL))
      }
      
      # Create API log directory if it doesn't exist
      api_log_dir <- file.path(self$log_dir, self$session_id, "api_logs")
      if (!dir.exists(api_log_dir)) {
        dir.create(api_log_dir, recursive = TRUE)
      }
      
      # Generate unique log file name for this API call
      timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S_%f")
      api_call_id <- sprintf("%s_%s_%s", provider, gsub("[^A-Za-z0-9_-]", "_", model), timestamp)
      
      # Create complete API log entry
      api_log_entry <- list(
        api_call_id = api_call_id,
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        session_id = self$session_id,
        provider = provider,
        model = model,
        request = list(
          prompt_content = prompt_content,
          metadata = request_metadata
        ),
        response = list(
          content = response_content,
          metadata = response_metadata
        ),
        call_sequence = self$performance_stats$api_calls + 1
      )
      
      # Write detailed API log to separate file
      api_log_file <- file.path(api_log_dir, sprintf("%s.json", api_call_id))
      writeLines(jsonlite::toJSON(api_log_entry, auto_unbox = TRUE, pretty = TRUE), api_log_file)
      
      # Also create human-readable version
      readable_file <- file.path(api_log_dir, sprintf("%s.md", api_call_id))
      readable_content <- sprintf(
        "# API Call: %s\n\n**Provider:** %s\n**Model:** %s\n**Timestamp:** %s\n**Session:** %s\n\n## Request Prompt\n\n```\n%s\n```\n\n## Response\n\n```\n%s\n```\n\n## Metadata\n\n**Request Metadata:**\n```json\n%s\n```\n\n**Response Metadata:**\n```json\n%s\n```\n",
        api_call_id,
        provider,
        model, 
        api_log_entry$timestamp,
        self$session_id,
        prompt_content,
        if(is.character(response_content)) paste(response_content, collapse = "\n") else as.character(response_content),
        if(is.null(request_metadata)) "null" else jsonlite::toJSON(request_metadata, auto_unbox = TRUE, pretty = TRUE),
        if(is.null(response_metadata)) "null" else jsonlite::toJSON(response_metadata, auto_unbox = TRUE, pretty = TRUE)
      )
      
      writeLines(readable_content, readable_file)
      
      # Log summary to main log
      self$debug(sprintf("API request/response logged: %s", api_call_id), list(
        api_call_id = api_call_id,
        provider = provider,
        model = model,
        prompt_length = nchar(as.character(prompt_content)),
        response_length = if(is.character(response_content)) sum(nchar(response_content)) else nchar(as.character(response_content)),
        files_created = c(
          sprintf("%s.json", api_call_id),
          sprintf("%s.md", api_call_id)
        )
      ))
      
      return(api_call_id)
    },
    
    #' @description
    #' Log cache operations
    #
    #
    #
    log_cache_operation = function(operation, key, size = NULL) {
      if (operation == "hit") {
        self$performance_stats$cache_hits <- self$performance_stats$cache_hits + 1
      } else if (operation == "miss") {
        self$performance_stats$cache_misses <- self$performance_stats$cache_misses + 1
      }
      
      context <- list(
        operation = operation,
        key = key,
        size_bytes = size,
        cache_hit_rate = round(self$performance_stats$cache_hits / 
                              (self$performance_stats$cache_hits + self$performance_stats$cache_misses), 3)
      )
      
      self$debug(sprintf("Cache %s: %s", operation, key), context)
    },
    
    #' @description
    #' Log cluster annotation progress
    #
    #
    #
    log_cluster_progress = function(cluster_id, stage, progress = NULL) {
      context <- list(
        cluster_id = cluster_id,
        stage = stage,
        progress = progress,
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      )
      
      self$info(sprintf("Cluster %s: %s", cluster_id, stage), context)
    },
    
    #' @description
    #' Log detailed cluster discussion with complete model conversations
    #
    #
    #
    log_discussion = function(cluster_id, event_type, data = NULL) {
      # Skip during R CMD check
      if (nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_", ""))) {
        # Still log to console/main log but skip file creation
        context <- list(
          cluster_id = cluster_id,
          event_type = event_type,
          data = data
        )
        self$info(sprintf("Discussion %s for cluster %s", event_type, cluster_id), context)
        return(invisible(NULL))
      }
      
      # Create session directory if needed
      session_dir <- file.path(self$log_dir, self$session_id)
      if (!dir.exists(session_dir)) {
        dir.create(session_dir, recursive = TRUE)
      }
      
      # Log to both main log and cluster-specific file
      context <- list(
        cluster_id = cluster_id,
        event_type = event_type,
        data = data
      )
      
      self$info(sprintf("Discussion %s for cluster %s", event_type, cluster_id), context)
      
      # Also create detailed cluster log file for easy access
      cluster_log_file <- file.path(session_dir, sprintf("cluster_%s_discussion.md", cluster_id))
      
      if (event_type == "start") {
        # Create markdown log for better readability
        content <- sprintf("# Cluster %s Discussion\n\n**Tissue:** %s\n**Markers:** %s\n**Started:** %s\n\n",
                          cluster_id, 
                          if(is.null(data$tissue_name)) "Unknown" else data$tissue_name,
                          if(is.null(data$marker_genes)) "Unknown" else data$marker_genes, 
                          format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
        cat(content, file = cluster_log_file)
        
      } else if (event_type == "prediction") {
        # Append model prediction in readable format
        # Handle both single strings and character vectors
        prediction_text <- if(is.vector(data$prediction) && length(data$prediction) > 1) {
          paste(data$prediction, collapse = "\n")
        } else {
          as.character(data$prediction)
        }
        
        content <- sprintf("## %s (Round %s)\n\n```\n%s\n```\n\n---\n\n",
                          data$model, 
                          if(is.null(data$round)) "1" else data$round, 
                          prediction_text)
        cat(content, file = cluster_log_file, append = TRUE)
        
      } else if (event_type == "consensus") {
        # Log consensus result
        content <- sprintf("## Consensus Result\n\n**Reached:** %s\n**Proportion:** %.2f\n**Entropy:** %.2f\n**Decision:** %s\n\n---\n\n",
                          data$reached %||% FALSE,
                          data$consensus_proportion %||% 0,
                          data$entropy %||% 0,
                          data$majority_prediction %||% "Unknown")
        cat(content, file = cluster_log_file, append = TRUE)
        
      } else if (event_type == "end") {
        # Final summary
        content <- sprintf("## Discussion Complete\n\n**Final Result:** %s\n**Completed:** %s\n",
                          data$final_result %||% "Unknown",
                          format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
        cat(content, file = cluster_log_file, append = TRUE)
      }
    },
    
    #' @description
    #' Get performance summary
    #
    get_performance_summary = function() {
      session_duration <- as.numeric(difftime(Sys.time(), self$performance_stats$session_start, units = "secs"))
      
      summary <- list(
        session_id = self$session_id,
        session_duration_seconds = round(session_duration, 1),
        api_calls = self$performance_stats$api_calls,
        cache_hits = self$performance_stats$cache_hits,
        cache_misses = self$performance_stats$cache_misses,
        cache_hit_rate = if (self$performance_stats$cache_hits + self$performance_stats$cache_misses > 0) {
          round(self$performance_stats$cache_hits / (self$performance_stats$cache_hits + self$performance_stats$cache_misses), 3)
        } else { 0 },
        errors = self$performance_stats$errors,
        warnings = self$performance_stats$warnings
      )
      
      self$info("Performance summary generated", summary)
      return(summary)
    },
    
    #' @description
    #' Clean up old log files
    #
    cleanup_logs = function(force = FALSE) {
      log_files <- list.files(self$log_dir, pattern = "^mllm_.*\\.log$", full.names = TRUE)
      
      if (length(log_files) == 0) return(invisible(NULL))
      
      # Sort by modification time (newest first)
      file_info <- file.info(log_files)
      log_files <- log_files[order(file_info$mtime, decreasing = TRUE)]
      
      files_to_remove <- c()
      
      if (force || length(log_files) > self$max_log_files) {
        # Remove excess files
        files_to_remove <- c(files_to_remove, log_files[(self$max_log_files + 1):length(log_files)])
      }
      
      # Check file sizes
      for (file in log_files[1:min(length(log_files), self$max_log_files)]) {
        size_mb <- file.info(file)$size / (1024 * 1024)
        if (size_mb > self$max_log_size) {
          files_to_remove <- c(files_to_remove, file)
        }
      }
      
      if (length(files_to_remove) > 0) {
        files_to_remove <- unique(files_to_remove)
        for (file in files_to_remove) {
          if (file.exists(file)) {
            file.remove(file)
          }
        }
        self$info(sprintf("Cleaned up %d log files", length(files_to_remove)))
      }
    },
    
    #' @description
    #' Set logging level
    #
    set_level = function(level) {
      old_level <- self$log_level
      self$log_level <- toupper(level)
      self$info(sprintf("Log level changed from %s to %s", old_level, self$log_level))
    }
  ),
  
  private = list(
    # Internal method to write log messages
    # @param level Log level
    # @param message Log message  
    # @param context Additional context
    log_message = function(level, message, context) {
      # Check if we should log this level
      level_priority <- list("DEBUG" = 1, "INFO" = 2, "WARN" = 3, "ERROR" = 4)
      if (level_priority[[level]] < level_priority[[self$log_level]]) {
        return(invisible(NULL))
      }
      
      timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      
      if (self$enable_json) {
        log_entry <- list(
          timestamp = timestamp,
          session_id = self$session_id,
          level = level,
          message = message,
          context = context,
          pid = Sys.getpid()
        )
        
        formatted_message <- jsonlite::toJSON(log_entry, auto_unbox = TRUE)
      } else {
        formatted_message <- sprintf("[%s] [%s] [%s] %s", 
                                   timestamp, self$session_id, level, message)
        if (!is.null(context)) {
          formatted_message <- paste(formatted_message, 
                                   sprintf("Context: %s", jsonlite::toJSON(context, auto_unbox = TRUE)))
        }
      }
      
      # Write to file
      private$write_to_file(formatted_message)
      
      # Output to console if enabled
      if (self$enable_console) {
        if (level == "ERROR") {
          cat(formatted_message, "\n", file = stderr())
        } else {
          cat(formatted_message, "\n")
        }
      }
    },
    
    # Write message to log file
    # @param message Formatted log message
    write_to_file = function(message) {
      # Skip file writing during R CMD check
      if (nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_", ""))) {
        return(invisible(NULL))
      }
      
      log_file <- file.path(self$log_dir, sprintf("mllm_%s.log", self$session_id))
      
      # Ensure log directory exists before writing
      if (!dir.exists(self$log_dir)) {
        dir.create(self$log_dir, recursive = TRUE)
      }
      
      # Check if log rotation is needed
      if (file.exists(log_file)) {
        size_mb <- file.info(log_file)$size / (1024 * 1024)
        if (size_mb > self$max_log_size) {
          # Rotate log file
          rotated_file <- file.path(self$log_dir, 
                                  sprintf("mllm_%s_%s.log", self$session_id, 
                                         format(Sys.time(), "%H%M%S")))
          file.rename(log_file, rotated_file)
        }
      }
      
      # Write to log file
      cat(message, "\n", file = log_file, append = TRUE)
    }
  )
)

#' Get the global logger instance
#
#' @export
get_logger <- function() {
  if (!exists(".mllm_logger", envir = .GlobalEnv) || is.null(.GlobalEnv$.mllm_logger)) {
    .GlobalEnv$.mllm_logger <- UnifiedLogger$new()
  }
  return(.GlobalEnv$.mllm_logger)
}

#' Set global logger configuration
#'
#' @param level Logging level: "DEBUG", "INFO", "WARN", or "ERROR". Default: "INFO"
#' @param console_output Whether to enable console output. Default: TRUE
#' @param json_format Whether to use JSON format for log messages. Default: TRUE
#'
#' @return Invisible logger object
#' @export
configure_logger <- function(level = "INFO", console_output = TRUE, json_format = TRUE) {
  logger <- get_logger()
  logger$set_level(level)
  logger$enable_console <- console_output
  logger$enable_json <- json_format
  
  invisible(logger)
}

#' Convenience functions for logging
#'
#' @param message Log message string
#' @param context Optional context information (list or character)
#'
#' @return Invisible NULL
#' @name logging_functions
NULL

#' @rdname logging_functions
#' @export
log_debug <- function(message, context = NULL) {
  get_logger()$debug(message, context)
}

#' @rdname logging_functions
#' @export
log_info <- function(message, context = NULL) {
  get_logger()$info(message, context)
}

#' @rdname logging_functions
#' @export
log_warn <- function(message, context = NULL) {
  get_logger()$warn(message, context)
}

#' @rdname logging_functions
#' @export
log_error <- function(message, context = NULL) {
  get_logger()$error(message, context)
}