# Load the RDS file
results <- readRDS('results/benchmark/reference_comparison/2_evaluation/lung_rare_results.rds')

# Examine the initial_results$consensus_results more deeply
cat("\n=== Initial Results Consensus Results (Detailed) ===\n")
for (cluster_id in names(results$initial_results$consensus_results)) {
  cat(sprintf("\nCluster %s consensus result:\n", cluster_id))
  consensus_result <- results$initial_results$consensus_results[[cluster_id]]
  # Print all fields and their values
  for (field in names(consensus_result)) {
    cat(sprintf("  %s: %s\n", field, paste(consensus_result[[field]], collapse=", ")))
  }

  # Check the class and structure
  cat(sprintf("  Class: %s\n", class(consensus_result)))
  cat("  Structure:\n")
  str(consensus_result)
}

# Check if there are any other fields in the results that might contain consensus information
cat("\n=== Other Potential Consensus Information ===\n")
all_fields <- names(results)
cat("Top-level fields in results:\n")
print(all_fields)

# Check if there's any discussion_logs$rounds$consensus_result
cat("\n=== Checking Discussion Logs for Consensus Results ===\n")
for (cluster_id in names(results$discussion_logs)) {
  cat(sprintf("\nCluster %s:\n", cluster_id))

  # Check if there are rounds
  if (length(results$discussion_logs[[cluster_id]]$rounds) > 0) {
    cat(sprintf("  Number of rounds: %d\n", length(results$discussion_logs[[cluster_id]]$rounds)))

    # Check each round for consensus_result
    for (round_idx in 1:length(results$discussion_logs[[cluster_id]]$rounds)) {
      round_data <- results$discussion_logs[[cluster_id]]$rounds[[round_idx]]
      cat(sprintf("  Round %d:\n", round_idx))
      cat(sprintf("    Fields: %s\n", paste(names(round_data), collapse=", ")))

      # If there's a consensus_result field, examine it
      if ("consensus_result" %in% names(round_data)) {
        cat("    Has consensus_result: YES\n")
        cat("    Consensus result structure:\n")
        str(round_data$consensus_result)
      } else {
        cat("    Has consensus_result: NO\n")
      }
    }
  } else {
    cat("  No rounds found\n")
  }
}

# Check if there's any information about how the final annotations were determined
cat("\n=== Final Annotations vs Initial Predictions ===\n")
for (cluster_id in names(results$final_annotations)) {
  cat(sprintf("\nCluster %s:\n", cluster_id))
  cat(sprintf("  Final annotation: %s\n", results$final_annotations[[cluster_id]]))

  # Get initial predictions for comparison
  if (!is.null(results$initial_results) &&
      !is.null(results$initial_results$individual_predictions)) {
    cat("  Initial predictions:\n")
    for (model in names(results$initial_results$individual_predictions)) {
      prediction <- results$initial_results$individual_predictions[[model]][cluster_id]
      cat(sprintf("    %s: %s\n", model, prediction))
    }
  }

  # Check if this matches any model's prediction exactly
  matches <- sapply(names(results$initial_results$individual_predictions), function(model) {
    prediction <- results$initial_results$individual_predictions[[model]][cluster_id]
    identical(prediction, results$final_annotations[[cluster_id]])
  })

  if (any(matches)) {
    cat(sprintf("  Matches prediction from: %s\n",
                paste(names(results$initial_results$individual_predictions)[matches], collapse=", ")))
  } else {
    cat("  Does not exactly match any initial prediction\n")
  }
}
