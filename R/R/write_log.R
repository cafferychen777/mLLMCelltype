#' Write Log Utility Function
#' @keywords internal
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