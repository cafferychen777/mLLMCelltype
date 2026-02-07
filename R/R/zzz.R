# Centralized global variable declarations for R CMD check
utils::globalVariables(c("custom_models", "cluster", "avg_log2FC", "gene"))

#' Package startup message
#' @keywords internal
.onAttach <- function(libname, pkgname) {
  # Get package version
  version <- utils::packageDescription(pkgname, fields = "Version")
  
  # ASCII art logo
  logo <- paste0(
    "\n",
    "+-------------------------------------------------------------------------+\n",
    "|                             mLLMCelltype                               |\n",
    "|                    Cell Type Annotation with LLMs                      |\n",
    "+-------------------------------------------------------------------------+\n"
  )
  
  # Create startup message
  msg <- paste0(
    logo,
    "\n",
    "mLLMCelltype v", version, " loaded successfully!\n",
    "For more information, please visit:\n",
    "- Documentation: https://cafferychen777.github.io/mLLMCelltype/\n",
    "- GitHub: https://github.com/cafferychen777/mLLMCelltype\n",
    "- Paper: https://doi.org/10.1101/2025.04.10.647852\n",
    "\n",
    "To cite this package in publications, use:\n",
    "  citation(\"mLLMCelltype\")"
  )
  
  
  # Display the message
  packageStartupMessage(msg)
  
  # Only show message when old cache is found
  old_cache <- file.path(".", "consensus_cache")
  if (dir.exists(old_cache)) {
    packageStartupMessage(
      "Note: Found old cache directory 'consensus_cache' in current directory.\n",
      "mLLMCelltype now uses system cache by default. To clean up:\n",
      "  unlink('consensus_cache', recursive = TRUE)\n",
      "For cache info, use: mllmcelltype_cache_dir()"
    )
  }
}

#' Package load message
#' @keywords internal
.onLoad <- function(libname, pkgname) {
  # Set any package-wide options here if needed
  # For example, you could set default logging level
  op <- options()
  op.mLLMCelltype <- list(
    mLLMCelltype.log_level = "INFO"
  )
  toset <- !(names(op.mLLMCelltype) %in% names(op))
  if(any(toset)) options(op.mLLMCelltype[toset])
  
  
  invisible()
}
