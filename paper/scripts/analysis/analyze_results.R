# Load the RDS file
results <- readRDS('results/benchmark/reference_comparison/2_evaluation/lung_rare_results.rds')

# Print structure of initial_results
cat("\n=== Initial Results Structure ===\n")
str(results$initial_results, max.level=2)

# Print individual predictions
cat("\n=== Individual Predictions ===\n")
print(results$initial_results$individual_predictions)

# Print consensus results
cat("\n=== Consensus Results ===\n")
print(results$initial_results$consensus_results)

# Print structure of discussion_logs
cat("\n=== Discussion Logs Structure ===\n")
for (i in 1:length(results$discussion_logs)) {
  cat(sprintf("\nCluster %d:\n", i))
  cat("  Initial predictions:\n")
  print(results$discussion_logs[[i]]$initial_predictions)

  cat("\n  Number of rounds:", length(results$discussion_logs[[i]]$rounds), "\n")

  # Check if there are rounds
  if (length(results$discussion_logs[[i]]$rounds) > 0) {
    # Get the last round
    last_round <- results$discussion_logs[[i]]$rounds[[length(results$discussion_logs[[i]]$rounds)]]

    # Print the names of elements in the last round
    cat("  Elements in the last round:\n")
    print(names(last_round))

    # Check if consensus_result exists in the last round
    if ("consensus_result" %in% names(last_round)) {
      cat("  Final consensus result:\n")
      print(last_round$consensus_result)
    } else {
      cat("  No consensus_result found in the last round\n")
    }
  } else {
    cat("  No rounds found\n")
  }
}

# Print final annotations
cat("\n=== Final Annotations ===\n")
print(results$final_annotations)
