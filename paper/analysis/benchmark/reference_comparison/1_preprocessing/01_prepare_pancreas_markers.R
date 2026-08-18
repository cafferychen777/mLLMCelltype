# Preprocessing script for Pancreas cell type benchmark
library(dplyr)
library(tidyr)
library(readr)

# Set paths
project_root <- "."

# Read the raw data as a character string
raw_data <- readLines(file.path(project_root, "data/reference/pancreas_markers_L1.csv"))

# Skip the header line and process each data line
data_lines <- raw_data[-1]  # Remove header line

# Process each line to extract cluster name and genes
markers_list <- list()
for (line in data_lines) {
  # Split the line into parts
  parts <- strsplit(line, ",")[[1]]

  # First part is the cluster name
  cluster_name <- parts[1]

  # Remaining parts are genes, trim any whitespace
  genes <- trimws(parts[-1])

  # Add to markers list
  markers_list[[cluster_name]] <- list(genes = genes)
}

# Print data structure for debugging
print("Processed data structure:")
str(markers_list)
print("\nNumber of clusters:")
print(length(markers_list))

# Print first and last cluster genes for verification
print("\nFirst cluster genes:")
print(markers_list[[1]]$genes)
print("\nLast cluster genes:")
print(markers_list[[length(markers_list)]]$genes)

# Create results directory if it doesn't exist
dir.create(file.path(project_root, "results/benchmark/reference_comparison/1_preprocessing"), recursive = TRUE, showWarnings = FALSE)

# Save preprocessed data
saveRDS(markers_list, file.path(project_root, "results/benchmark/reference_comparison/1_preprocessing/pancreas_markers.rds"))

print("\nPreprocessing complete. Data saved to pancreas_markers.rds")
