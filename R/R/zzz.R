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