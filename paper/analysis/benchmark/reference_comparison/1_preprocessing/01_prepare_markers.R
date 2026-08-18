# Generic preprocessing script for tissue cell type benchmark
library(dplyr)
library(tidyr)
library(readr)

preprocess_markers <- function(dataset_name) {
  # Set paths
  project_root <- "."

  # Construct input file path
  input_file <- file.path(project_root,
                         paste0("data/reference/", dataset_name, "_markers.csv"))

  # Read the raw data as a character string
  raw_data <- readLines(input_file)

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
  output_dir <- file.path(project_root, "results/benchmark/reference_comparison/1_preprocessing")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Save preprocessed data
  output_file <- file.path(output_dir, paste0(dataset_name, "_markers.rds"))
  saveRDS(markers_list, output_file)

  print(paste0("\nPreprocessing complete. Data saved to ", dataset_name, "_markers.rds"))
}

preprocess_markers("lung_rare")
