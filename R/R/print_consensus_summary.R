#' Print summary of consensus results
#' 
#' This function prints a detailed summary of the consensus building process,
#' including initial predictions from all models, uncertainty metrics, and final consensus
#' for each controversial cluster.
#' 
#
#'   \itemize{
#'     \item initial_results: A list containing individual_predictions, consensus_results, and controversial_clusters
#'     \item final_annotations: A list of final cell type annotations for each cluster
#'     \item controversial_clusters: A character vector of cluster IDs that were controversial
#'     \item discussion_logs: A list of discussion logs for each controversial cluster
#'   }
#
#' @keywords internal
print_consensus_summary <- function(results) {
  to_scalar_or_na <- function(x) {
    if (is.null(x) || length(x) == 0) return(NA_character_)
    scalar <- tryCatch(as.character(x[[1]]), error = function(e) NA_character_)
    if (is.na(scalar) || !nzchar(trimws(scalar))) return(NA_character_)
    scalar
  }

  to_display_text <- function(x, fallback = "No prediction provided") {
    scalar <- to_scalar_or_na(x)
    if (is.na(scalar)) fallback else scalar
  }

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

          cat(sprintf("  %s: %s\n", model, to_display_text(prediction)))
        }
      } 
      # If no initial predictions in discussion logs, use initial_results
      else if (!is.null(results$initial_results) &&
               !is.null(results$initial_results$individual_predictions)) {
        # Derive sorted cluster order for positional fallback
        all_cluster_ids <- names(results$final_annotations)

        # Check naming convention once using the first model's predictions
        first_model_preds <- results$initial_results$individual_predictions[[1]]
        has_names <- !is.null(names(first_model_preds))

        for (model in names(results$initial_results$individual_predictions)) {
          model_preds <- results$initial_results$individual_predictions[[model]]

          if (has_names) {
            prediction <- model_preds[[char_cluster_id]]
          } else {
            # Positional fallback: find cluster's position among all clusters
            r_index <- match(char_cluster_id, all_cluster_ids)
            prediction <- if (!is.na(r_index) && r_index >= 1 && r_index <= length(model_preds)) {
              model_preds[r_index]
            } else {
              NA
            }
          }

          cat(sprintf("  %s: %s\n", model, to_display_text(prediction)))
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
        
        final_annotation_str <- to_scalar_or_na(final_annotation)
        if (is.na(final_annotation_str)) {
          final_annotation_str <- "Final_Annotation_Missing"
        }
        
        # Validate consistency between final consensus and initial predictions
        if (!is.null(results$initial_results) && 
            !is.null(results$initial_results$individual_predictions) &&
            length(names(results$initial_results$individual_predictions)) > 0) { # Ensure there are models
          
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
              pred_scalar <- to_scalar_or_na(pred)
              if (!is.na(pred_scalar) && pred_scalar != "No prediction provided") {
                all_predictions[[model]] <- pred_scalar
              }
            } # End model loop

            # Check if all collected models predicted the same result and compare with final consensus
            if (length(all_predictions) > 0) {
              unique_preds <- unique(unlist(all_predictions))
              if (length(unique_preds) == 1) {
                # We don't need to clean prefix here as discussion log preds should be clean
                clean_pred <- trimws(unique_preds[1]) # Just trim whitespace
                
                # If all models predicted same but differs from final consensus, add warning
                # Use normalize_annotation for robust comparison (handles punctuation, plurals, synonyms)
                if (normalize_annotation(clean_pred) != normalize_annotation(final_annotation_str)) {
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
