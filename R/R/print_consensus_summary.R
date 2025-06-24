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
      # Ensure cluster_id is always a string type
      char_cluster_id <- as.character(cluster_id)
      
      cat(sprintf("\nCluster %s:\n", char_cluster_id))
      cat("Initial predictions:\n")
      
      # Prioritize initial predictions from discussion_logs, as these are the actual predictions used for discussion
      if (!is.null(results$discussion_logs) && 
          !is.null(results$discussion_logs[[char_cluster_id]]) && 
          !is.null(results$discussion_logs[[char_cluster_id]]$initial_predictions)) {
        
        # Use initial predictions from discussion logs
        initial_predictions <- results$discussion_logs[[char_cluster_id]]$initial_predictions
        
        # Iterate through each model's prediction
        for (model in names(initial_predictions)) {
          prediction <- initial_predictions[[model]]
          
          # Handle empty or NA predictions
          if (is.null(prediction) || is.na(prediction) || prediction == "") {
            prediction <- "No prediction provided"
          }
          
          cat(sprintf("  %s: %s\n", model, prediction))
        }
      } 
      # If no initial predictions in discussion logs, use initial_results
      else if (!is.null(results$initial_results) && 
               !is.null(results$initial_results$individual_predictions)) {
        # Iterate through each model's prediction
        for (model in names(results$initial_results$individual_predictions)) {
          # Check if prediction has names
          first_model <- names(results$initial_results$individual_predictions)[1]
          predictions <- results$initial_results$individual_predictions[[first_model]]
          has_names <- !is.null(names(predictions))
          
          if (has_names) {
            # If it has names, use string indexing
            if (char_cluster_id %in% names(results$initial_results$individual_predictions[[model]])) {
              prediction <- results$initial_results$individual_predictions[[model]][[char_cluster_id]]
            } else {
              prediction <- NA
            }
          } else {
            # Even without a name, try to use string indexing
            # First try to convert the string to a numeric value as an index
            cluster_idx <- as.numeric(char_cluster_id)
            if (!is.na(cluster_idx) && cluster_idx <= length(results$initial_results$individual_predictions[[model]])) {
              prediction <- results$initial_results$individual_predictions[[model]][cluster_idx]
            } else {
              prediction <- NA
            }
          }
          
          # Handle empty or NA predictions
          if (is.null(prediction) || is.na(prediction) || prediction == "") {
            prediction <- "No prediction provided"
          }
          
          cat(sprintf("  %s: %s\n", model, prediction))
        }
      } else {
        cat("  No initial predictions available\n")
      }
      
      # Print uncertainty metrics
      cat("\nUncertainty metrics:\n")
      if (!is.null(results$initial_results) && 
          !is.null(results$initial_results$consensus_results) && 
          !is.null(results$initial_results$consensus_results[[char_cluster_id]])) {
        
        consensus_result <- results$initial_results$consensus_results[[char_cluster_id]]
        
        # Print consensus proportion
        if (!is.null(consensus_result$consensus_proportion)) {
          cat(sprintf("  Consensus proportion: %.2f\n", consensus_result$consensus_proportion))
        }
        
        # Print Shannon entropy
        if (!is.null(consensus_result$entropy)) {
          cat(sprintf("  Shannon entropy: %.2f\n", consensus_result$entropy))
        }
      } else {
        cat("  Uncertainty metrics not available\n")
      }
      
      # Print final consensus
      if (!is.null(results$final_annotations) && 
          !is.null(results$final_annotations[[char_cluster_id]])) {
        
        final_annotation <- results$final_annotations[[char_cluster_id]]
        
        # Handle NA values first
        if (is.null(final_annotation) || is.na(final_annotation)) {
          final_annotation_str <- "Final_Annotation_Missing"
        } else if (is.list(final_annotation) || (is.vector(final_annotation) && length(final_annotation) > 0 && !is.character(final_annotation))) {
          # If it's a list or non-character vector, take the first element
          final_annotation_str <- tryCatch({
            as.character(final_annotation[[1]])
          }, error = function(e) {
            "Error_Converting_To_String"
          })
        } else {
          # Otherwise convert directly to string
          final_annotation_str <- tryCatch({
            as.character(final_annotation)
          }, error = function(e) {
            "Error_Converting_To_String"
          })
        }
        
        # Validate consistency between final consensus and initial predictions
        if (!is.null(results$initial_results) && 
            !is.null(results$initial_results$individual_predictions) &&
            length(names(results$initial_results$individual_predictions)) > 0) { # Ensure there are models
          
          # Try to get the first model name
          tryCatch({
            first_model <- names(results$initial_results$individual_predictions)[1]
            predictions <- results$initial_results$individual_predictions[[first_model]]
            has_names <- all(names(results$initial_results$individual_predictions) %in% names(results$discussion_logs[[char_cluster_id]]$initial_predictions))
          }, error = function(e) {
            has_names <- FALSE
          })
          
          # Collect predictions from all models
          all_predictions <- list()
          # Validation using discussion log predictions
          discussion_log_predictions <- NULL
          if (!is.null(results$discussion_logs) && 
              !is.null(results$discussion_logs[[char_cluster_id]]) && 
              !is.null(results$discussion_logs[[char_cluster_id]]$initial_predictions)) {
            discussion_log_predictions <- results$discussion_logs[[char_cluster_id]]$initial_predictions
          }

          if (!is.null(discussion_log_predictions) && length(discussion_log_predictions) > 0) {
            all_predictions <- list()
            
            for (model in names(discussion_log_predictions)) {
              pred <- discussion_log_predictions[[model]]
              
              # Check if prediction is valid and not the placeholder
              if (!is.null(pred)) {
                if (!is.na(pred)) {
                  # Make sure to compare against the placeholder string too
                  if (pred != "" && pred != "No prediction provided") { 
                     all_predictions[[model]] <- pred
                  }
                }
              }
            } # End model loop

            # Check if all collected models predicted the same result and compare with final consensus
            if (length(all_predictions) > 0) {
              unique_preds <- unique(unlist(all_predictions))
              if (length(unique_preds) == 1) {
                # We don't need to clean prefix here as discussion log preds should be clean
                clean_pred <- trimws(unique_preds[1]) # Just trim whitespace
                
                # If all models predicted same but differs from final consensus, add warning
                # Use grepl for substring matching robustness
                if (!grepl(clean_pred, final_annotation_str, ignore.case = TRUE) && 
                    !grepl(final_annotation_str, clean_pred, ignore.case = TRUE)) {
                  cat(sprintf("WARNING: All models in discussion log predicted '%s' but final consensus is '%s'\n", 
                              clean_pred, final_annotation_str))
                }
              }
            }
          } # End check for discussion_log_predictions
        }
        
        cat(sprintf("Final consensus: %s\n", final_annotation_str))
      } else {
        cat("Final consensus: Not available\n")
      }
    }
  }
  
  # If no controversial clusters, print message
  if (length(results$controversial_clusters) == 0) {
    cat("\nNo controversial clusters found. All clusters reached consensus.\n")
  }
}