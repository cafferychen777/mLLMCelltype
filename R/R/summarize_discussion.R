#' Summarize discussion and determine final cell type
#' 
#' NOTE: This function is currently not in use. The consensus_annotation.R file
#' now directly extracts the majority_prediction from the last round of discussion.
#' This function is kept for potential future use or reference.
#'
#' @param discussion_log Discussion log for a cluster
#' @param cluster_id Cluster identifier
#' @param model Model to use for summary
#' @param api_key API key for the model
#' @return Final cell type determination
#' @keywords internal
summarize_discussion <- function(discussion_log, cluster_id, model, api_key) {
  # Check if discussion log is empty
  if (length(discussion_log) == 0 || all(nchar(trimws(discussion_log)) == 0)) {
    message("No discussion log found for cluster ", cluster_id, ". Skipping summary.")
    return("Controversial")
  }
  
  # Format the discussion log for the model
  formatted_discussion <- paste(
    "Based on the following discussion, determine the final cell type.",
    "ONLY output the cell type name, without any numbers, prefixes, or analysis.",
    "The cell type can be a mixture of multiple cell types if necessary.",
    "If one model says 'CELL TYPE: NK cells' with detailed reasoning and another just says 'Natural Killer cells', they should be considered the same type.",
    "",
    "Example good outputs:",
    "NK cells",
    "T cells",
    "NK and T cells",
    "Natural Killer T cells",
    "",
    "Example bad outputs:",
    "1. NK cells",
    "2: Natural Killer cells",
    "cluster1: NK cells",
    "CELL TYPE: NK cells",
    "Based on the discussion, the cell type is NK cells",
    "",
    "Discussion log:",
    paste(discussion_log, collapse = "\n"),
    sep = "\n"
  )
  
  # Get summary from model
  response <- get_model_response(formatted_discussion, model, api_key)
  
  # Extract cell type (first line) and summary (rest)
  lines <- strsplit(response, "\n")[[1]]
  cell_type <- trimws(lines[1])  # First line is the cell type
  summary <- paste(lines[-1], collapse = "\n")  # Rest is the summary
  
  # Clean up cell type by removing common prefixes and formatting
  # Remove numbered prefixes like "1. ", "1: ", "1- ", etc.
  cell_type <- gsub("^\\d+[\\.:\\-\\s]+\\s*", "", cell_type)
  
  # Remove "cluster" prefixes like "cluster1:", "cluster 1:", etc.
  cell_type <- gsub("^cluster\\s*\\d+[\\s:]*", "", cell_type)
  
  # Remove "CELL TYPE:" prefix
  cell_type <- gsub("^CELL\\s*TYPE[\\s:]*", "", cell_type)
  
  # Remove any remaining prefixes like "1. 1: "
  cell_type <- gsub("^[\\d\\.:\\-\\s]+", "", cell_type)
  
  # Final trim of whitespace
  cell_type <- trimws(cell_type)
  
  # Check if cell type is actually an error message
  if (grepl("no discussion|cannot determine", tolower(cell_type))) {
    message("Model returned an error message instead of cell type. Marking as controversial.")
    return("Controversial")
  }
  
  # Log final consensus
  log_info("Final consensus determined", list(
    cluster_id = cluster_id,
    cell_type = cell_type,
    model = model,
    summary_length = nchar(summary)
  ))
  
  cell_type  # Return just the cell type
}