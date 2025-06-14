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
    #' @param base_dir Base directory for logs (default: "logs")
    #' @param level Logging level: DEBUG, INFO, WARN, ERROR (default: "INFO")
    #' @param max_size Maximum log file size in MB (default: 10)
    #' @param max_files Maximum number of log files to keep (default: 5)
    #' @param console_output Whether to output to console (default: TRUE)
    #' @param json_format Whether to use JSON format (default: TRUE)
    initialize = function(base_dir = "logs", level = "INFO", max_size = 10, 
                         max_files = 5, console_output = TRUE, json_format = TRUE) {
      self$log_dir <- base_dir
      self$log_level <- toupper(level)
      self$max_log_size <- max_size
      self$max_log_files <- max_files
      self$enable_console <- console_output
      self$enable_json <- json_format
      
      # Create log directory if it doesn't exist
      if (!dir.exists(self$log_dir)) {
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
    #' @param message Log message
    #' @param context Additional context (optional)
    debug = function(message, context = NULL) {
      private$log_message("DEBUG", message, context)
    },
    
    #' @description
    #' Log an info message
    #' @param message Log message
    #' @param context Additional context (optional)
    info = function(message, context = NULL) {
      private$log_message("INFO", message, context)
    },
    
    #' @description
    #' Log a warning message
    #' @param message Log message
    #' @param context Additional context (optional)
    warn = function(message, context = NULL) {
      self$performance_stats$warnings <- self$performance_stats$warnings + 1
      private$log_message("WARN", message, context)
    },
    
    #' @description
    #' Log an error message
    #' @param message Log message
    #' @param context Additional context (optional)
    error = function(message, context = NULL) {
      self$performance_stats$errors <- self$performance_stats$errors + 1
      private$log_message("ERROR", message, context)
    },
    
    #' @description
    #' Log API call performance
    #' @param provider API provider name
    #' @param model Model name
    #' @param duration Duration in seconds
    #' @param success Whether the call was successful
    #' @param tokens Number of tokens used (optional)
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
    #' Log cache operations
    #' @param operation Operation type: "hit", "miss", "store", "clear"
    #' @param key Cache key
    #' @param size Size of cached data (optional)
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
    #' @param cluster_id Cluster identifier
    #' @param stage Current stage
    #' @param progress Progress information
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
    #' Get performance summary
    #' @return List of performance statistics
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
    #' @param force Force cleanup even if within file limits
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
    #' @param level New logging level: DEBUG, INFO, WARN, ERROR
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
      log_file <- file.path(self$log_dir, sprintf("mllm_%s.log", self$session_id))
      
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
#' @return UnifiedLogger instance
#' @export
get_logger <- function() {
  if (!exists(".mllm_logger", envir = .GlobalEnv) || is.null(.GlobalEnv$.mllm_logger)) {
    .GlobalEnv$.mllm_logger <- UnifiedLogger$new()
  }
  return(.GlobalEnv$.mllm_logger)
}

#' Set global logger configuration
#' @param level Logging level
#' @param console_output Whether to output to console
#' @param json_format Whether to use JSON format
#' @export
configure_logger <- function(level = "INFO", console_output = TRUE, json_format = TRUE) {
  logger <- get_logger()
  logger$set_level(level)
  logger$enable_console <- console_output
  logger$enable_json <- json_format
  
  invisible(logger)
}

#' Convenience functions for logging
#' @param message Log message
#' @param context Additional context (optional)
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