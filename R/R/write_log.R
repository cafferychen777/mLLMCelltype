#' Write Log Utility Function
#'
#' This function writes log messages to a file with timestamps and also prints them to the console.
#'
#' @param message Character string containing the log message to write
#' @param log_file Character string specifying the log file name (default: "llm_celltype.log")
#' @return None, used for side effects
#' @export
write_log <- function(message, log_file = "llm_celltype.log") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  log_entry <- sprintf("[%s] %s\n", timestamp, message)
  
  # Get the current working directory
  current_dir <- getwd()
  
  # Ensure log directory exists
  log_dir <- file.path(current_dir, "logs")
  if (!dir.exists(log_dir)) {
    dir.create(log_dir)
  }
  
  # Write to log
  log_path <- file.path(log_dir, log_file)
  cat(log_entry, file = log_path, append = TRUE)
  
  # Also print to console for debugging
  cat(log_entry)
}