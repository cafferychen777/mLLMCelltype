# Summarize PopV comparison results across datasets
# This script reads the individual dataset popV comparison results and summarizes them

library(tidyverse)

# Define the results directory
results_dir <- "results/benchmark/popv_comparison/2_evaluation"

# Initialize summary data frame
summary_df <- data.frame(
  Dataset = character(),
  Tissue = character(),
  mLLMCelltype_Accuracy = numeric(),
  PopV_Accuracy = numeric(),
  Improvement = numeric(),
  stringsAsFactors = FALSE
)

# Function to calculate accuracy from match results
calculate_accuracy <- function(match_col) {
  # Scoring: Exact match = 1.0, Parent/Sibling/Child match = 0.5, No match = 0
  scores <- case_when(
    grepl("Exact", match_col, ignore.case = TRUE) ~ 1.0,
    grepl("Parent|Sibling|Child", match_col, ignore.case = TRUE) ~ 0.5,
    grepl("No match", match_col, ignore.case = TRUE) ~ 0,
    TRUE ~ 0
  )
  mean(scores, na.rm = TRUE) * 100
}

cat("Summarizing PopV comparison results across datasets...\n\n")

# 1. Thymus
thymus_file <- file.path(results_dir, "Thymus_results_with_match.csv")
if (file.exists(thymus_file)) {
  thymus_data <- read_csv(thymus_file, show_col_types = FALSE)
  cat("Found Thymus dataset with", nrow(thymus_data), "clusters\n")

  # You may need to adjust column names based on actual file structure
  # Placeholder - needs to be checked
}

# 2. HLCA / Lung
hlca_file <- file.path(results_dir, "HLCA_leiden_4_results.csv")
if (file.exists(hlca_file)) {
  hlca_data <- read_csv(hlca_file, show_col_types = FALSE)
  cat("Found HLCA dataset with", nrow(hlca_data), "clusters\n")
}

# 3. LCA
lca_file <- file.path(results_dir, "LCA_popv_fast_results.csv")
if (file.exists(lca_file)) {
  cat("Found LCA dataset\n")
  # Process LCA results
}

# 4-7. Other datasets (PBMC, Intestine, etc.)
other_files <- list.files(results_dir, pattern = ".*_results.*\\.csv$", full.names = TRUE)
cat("\nFound", length(other_files), "total result files\n")
cat("Files:\n")
cat(paste("  -", basename(other_files), "\n"))

cat("\n=== Summary ===\n")
cat("To complete the PopV comparison summary, you need to:\n")
cat("1. Check the actual column names in each CSV file\n")
cat("2. Calculate accuracy for mLLMCelltype and PopV using the match columns\n")
cat("3. Generate Extended Data Table with the summary statistics\n\n")

cat("Based on manuscript text (sn-article.tex):\n")
cat("  - Thymus: mLLMCelltype improved by +10.6% over popV\n")
cat("  - LCA: mLLMCelltype improved by +13.17% over popV\n\n")

cat("These can serve as benchmarks for validating the calculated results.\n")
