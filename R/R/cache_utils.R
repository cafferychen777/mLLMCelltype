#' Get mLLMCelltype cache location
#' 
#' @description Display the cache directory location
#'
#' @param cache_dir Cache directory specification. NULL uses system default, "local" uses current dir, "temp" uses temp dir, or custom path
#'
#' @return Invisible cache directory path
#' @export
#' @examples
#' \dontrun{
#' mllmcelltype_cache_dir()
#' mllmcelltype_cache_dir("local")
#' }
mllmcelltype_cache_dir <- function(cache_dir = NULL) {
  # Create temporary cache_manager to get actual path
  cm <- CacheManager$new(cache_dir)
  actual_path <- cm$get_cache_dir()
  
  message("Cache directory: ", actual_path)
  return(invisible(actual_path))
}

#' Clear mLLMCelltype cache
#' 
#' @description Clear the mLLMCelltype cache
#'
#' @param cache_dir Cache directory specification. NULL uses system default, "local" uses current dir, "temp" uses temp dir, or custom path
#'
#' @return Invisible NULL
#' @export
#' @examples
#' \dontrun{
#' mllmcelltype_clear_cache()
#' mllmcelltype_clear_cache("local")
#' }
mllmcelltype_clear_cache <- function(cache_dir = NULL) {
  # Create temporary cache_manager to get actual path
  cm <- CacheManager$new(cache_dir)
  actual_path <- cm$get_cache_dir()
  
  if (!dir.exists(actual_path)) {
    message("Cache directory does not exist: ", actual_path)
    return(invisible(NULL))
  }
  
  cache_files <- list.files(actual_path, pattern = "\\.rds$", full.names = TRUE)
  
  if (length(cache_files) == 0) {
    message("No cache files found in: ", actual_path)
    return(invisible(NULL))
  }
  
  if (interactive()) {
    response <- readline(prompt = sprintf(
      "Delete %d cache files in %s? (yes/no): ",
      length(cache_files), actual_path
    ))
    
    if (!tolower(response) %in% c("yes", "y")) {
      message("Cache clearing cancelled.")
      return(invisible(NULL))
    }
  }
  
  unlink(cache_files)
  message(sprintf("Cleared %d cache files from: %s", length(cache_files), actual_path))
  
  invisible(NULL)
}