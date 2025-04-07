#' Print summary of consensus results
#' 
#' This function prints a detailed summary of the consensus building process,
#' including initial predictions from all models, uncertainty metrics, and final consensus
#' for each controversial cluster.
#' 
#' @param results A list containing consensus annotation results with the following components:
#'   \itemize{
#'     \item initial_results: A list containing individual_predictions, consensus_results, and controversial_clusters
#'     \item final_annotations: A list of final cell type annotations for each cluster
#'     \item controversial_clusters: A character vector of cluster IDs that were controversial
#'     \item discussion_logs: A list of discussion logs for each controversial cluster
#'   }
#' @return None, prints summary to console
#' @keywords internal
print_consensus_summary <- function(results) {
  # Print consensus building summary
  cat("\nConsensus Building Summary:\n")
  cat(sprintf("Total clusters analyzed: %d\n", length(results$final_annotations)))
  cat(sprintf("Controversial clusters requiring discussion: %d\n", 
              length(results$controversial_clusters)))
  
  # If there are controversial clusters, print detailed results
  if (length(results$controversial_clusters) > 0) {
    cat("\nDetailed results for controversial clusters:\n")
    
    # Iterate through each controversial cluster
    for (cluster_id in results$controversial_clusters) {
      char_cluster_id <- as.character(cluster_id)
      numeric_cluster_id <- as.numeric(cluster_id)
      
      cat(sprintf("\nCluster %s:\n", char_cluster_id))
      cat("Initial predictions:\n")
      
      # Print initial predictions from each model
      if (!is.null(results$initial_results) && 
          !is.null(results$initial_results$individual_predictions)) {
        
        # Check if predictions have names
        first_model <- names(results$initial_results$individual_predictions)[1]
        predictions <- results$initial_results$individual_predictions[[first_model]]
        has_names <- !is.null(names(predictions))
        
        # Iterate through each model, get and print its prediction
        for (model in names(results$initial_results$individual_predictions)) {
          prediction <- tryCatch({
            # Access prediction differently based on whether it has names
            if (has_names) {
              pred <- results$initial_results$individual_predictions[[model]][char_cluster_id]
            } else {
              pred <- results$initial_results$individual_predictions[[model]][numeric_cluster_id]
            }
            
            # Handle empty or NA predictions
            if (is.null(pred) || is.na(pred) || pred == "") {
              "Unknown"
            } else {
              pred
            }
          }, error = function(e) {
            # Handle errors when retrieving prediction
            "Error retrieving prediction"
          })
          
          cat(sprintf("  %s: %s\n", model, prediction))
        }
      } else {
        cat("  Initial predictions not available\n")
      }
      
      # Print uncertainty metrics
      if (!is.null(results$initial_results) && 
          !is.null(results$initial_results$consensus_results) &&
          !is.null(results$initial_results$consensus_results[[char_cluster_id]])) {
        
        consensus_result <- results$initial_results$consensus_results[[char_cluster_id]]
        
        cat("\nUncertainty metrics:\n")
        
        # Print consensus proportion
        if (!is.null(consensus_result$consensus_proportion)) {
          cat(sprintf("  Consensus proportion: %.2f\n", consensus_result$consensus_proportion))
        } else if (!is.null(consensus_result$agreement_score)) {
          # If no consensus_proportion but agreement_score exists, use it as alternative
          cat(sprintf("  Agreement score: %.2f\n", consensus_result$agreement_score))
        }
        
        # Print entropy
        if (!is.null(consensus_result$entropy)) {
          cat(sprintf("  Shannon entropy: %.2f\n", consensus_result$entropy))
        }
      } else {
        cat("\nUncertainty metrics: Not available\n")
      }
      
      # Print final consensus
      if (!is.null(results$final_annotations) && 
          !is.null(results$final_annotations[[char_cluster_id]])) {
        
        final_annotation <- results$final_annotations[[char_cluster_id]]
        
        # Process final_annotation differently based on its type
        if (is.list(final_annotation) || (is.vector(final_annotation) && length(final_annotation) > 0 && !is.character(final_annotation))) {
          # If it's a list or non-character vector, take the first element
          final_annotation_str <- as.character(final_annotation[[1]])
        } else {
          # Otherwise convert directly to string
          final_annotation_str <- as.character(final_annotation)
        }
        
        cat(sprintf("Final consensus: %s\n", final_annotation_str))
      } else {
        cat("Final consensus: Not available\n")
      }
    }
  } else {
    cat("\nNo controversial clusters found. All clusters reached consensus.\n")
  }
}
