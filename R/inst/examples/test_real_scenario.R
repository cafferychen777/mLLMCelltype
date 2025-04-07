# Script to test interactive_consensus_annotation in a real-world scenario

# Load required packages
library(devtools)
library(ggplot2)

# Load dotenv package
if (!requireNamespace("dotenv", quietly = TRUE)) {
  install.packages("dotenv")
}
library(dotenv)

# Set working directory to the package root
setwd("/Users/apple/Research/LLMCelltype/code/R")

# Load .env file
project_root <- "/Users/apple/Research/LLMCelltype"
dotenv::load_dot_env(file.path(project_root, ".env"))

# Verify API keys are loaded
cat("QWEN_API_KEY loaded:", nchar(Sys.getenv("QWEN_API_KEY")) > 0, "\n")
cat("ANTHROPIC_API_KEY loaded:", nchar(Sys.getenv("ANTHROPIC_API_KEY")) > 0, "\n")
cat("OPENAI_API_KEY loaded:", nchar(Sys.getenv("OPENAI_API_KEY")) > 0, "\n")
cat("GEMINI_API_KEY loaded:", nchar(Sys.getenv("GEMINI_API_KEY")) > 0, "\n")

# Load API keys from environment variables
api_keys <- list(
  # Model names as keys
  "claude-3-5-sonnet-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
  "gemini-1.5-pro" = Sys.getenv("GEMINI_API_KEY"),
  "qwen-max-2025-01-25" = Sys.getenv("QWEN_API_KEY"),
  "gpt-4o" = Sys.getenv("OPENAI_API_KEY"),
  
  # Provider names as keys (needed by some functions)
  "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
  "gemini" = Sys.getenv("GEMINI_API_KEY"),
  "qwen" = Sys.getenv("QWEN_API_KEY"),
  "openai" = Sys.getenv("OPENAI_API_KEY")
)

# Create more realistic mock data for a PBMC dataset
create_realistic_mock_data <- function() {
  # Create a more comprehensive list of genes for each cluster
  data <- list(
    "1" = list(
      genes = c("CD19", "CD20", "CD79A", "CD79B", "PAX5", "MS4A1", "CD22", "CD24", "BANK1", "BLNK", 
               "FCRL1", "FCRL2", "FCRL5", "HLA-DRA", "HLA-DRB1"),
      description = "B cell markers"
    ),
    "2" = list(
      genes = c("CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "IL7R", "CCR7", "LEF1", "TCF7", 
               "SELL", "CD27", "CD28", "TRAC", "TRBC1"),
      description = "T cell markers"
    ),
    "3" = list(
      genes = c("CD14", "CD68", "CD163", "CSF1R", "ITGAM", "FCGR1A", "FCGR3A", "S100A8", "S100A9", "LYZ", 
               "VCAN", "MNDA", "MS4A7", "CYBB", "CLEC7A"),
      description = "Monocyte markers"
    ),
    "4" = list(
      genes = c("NCAM1", "NKG7", "KLRD1", "KLRB1", "KLRC1", "KLRC2", "KLRF1", "FCGR3A", "FCGR3B", "GZMB", 
               "GZMH", "GNLY", "PRF1", "FCER1G", "SLAMF7"),
      description = "NK cell markers"
    ),
    "5" = list(
      genes = c("CD1C", "CLEC10A", "CLEC4C", "FCER1A", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "IRF4", "NRP1", 
               "BATF3", "ZBTB46", "ITGAX", "ITGAM", "CD86"),
      description = "Dendritic cell markers"
    ),
    "6" = list(
      genes = c("PPBP", "PF4", "ITGA2B", "GP1BA", "GP9", "ITGB3", "SELP", "PEAR1", "MPL", "THBS1", 
               "F13A1", "TUBB1", "GP6", "MMRN1", "TREML1"),
      description = "Platelet markers"
    ),
    "7" = list(
      genes = c("HBB", "HBA1", "HBA2", "ALAS2", "AHSP", "CA1", "CA2", "SLC4A1", "GYPA", "GYPB", 
               "GYPC", "ANK1", "EPB42", "SPTA1", "SPTB"),
      description = "Erythrocyte markers"
    )
  )
  return(data)
}

# Create a temporary directory for logs
temp_log_dir <- tempfile("test_logs")
dir.create(temp_log_dir)
cat("Created temporary log directory:", temp_log_dir, "\n")

# Create a temporary directory for cache
temp_cache_dir <- tempfile("test_cache")
dir.create(temp_cache_dir)
cat("Created temporary cache directory:", temp_cache_dir, "\n")

# Load the package
cat("\nLoading package...\n")
load_all(".")

# Run the test with realistic data
cat("\nTesting interactive_consensus_annotation with realistic data...\n")

# Create realistic mock data
mock_data <- create_realistic_mock_data()

# Set tissue name
tissue_name <- "Human PBMC"

# Set models to use (use multiple models for a more realistic test)
test_models <- c(
  "claude-3-5-sonnet-latest",  # Anthropic
  "gemini-1.5-pro",           # Google
  "qwen-max-2025-01-25",      # Qwen
  "gpt-4o"                    # OpenAI
)

# Run the refactored function with realistic data
result <- tryCatch({
  # Use the refactored function
  interactive_consensus_annotation(
    input = mock_data,
    tissue_name = tissue_name,
    models = test_models,
    api_keys = api_keys,
    top_gene_count = 10,
    controversy_threshold = 0.7,
    entropy_threshold = 1.0,
    max_discussion_rounds = 2,
    log_dir = temp_log_dir,
    cache_dir = temp_cache_dir,
    use_cache = TRUE
  )
}, error = function(e) {
  cat("Error occurred during annotation:", e$message, "\n")
  return(e)
})

# Check if the result is an error
if (inherits(result, "error")) {
  cat("\nTest failed with error!\n")
} else {
  cat("\nTest succeeded!\n")
  
  # Print the results
  cat("\nFinal annotations:\n")
  print(result$final_annotations)
  
  # Check if log files were created
  log_files <- list.files(temp_log_dir)
  cat("\nLog files created:", length(log_files), "\n")
  if (length(log_files) > 0) {
    print(log_files)
  }
  
  # Create a visualization of the results
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    # Create a data frame for plotting
    plot_data <- data.frame(
      Cluster = names(result$final_annotations),
      CellType = unlist(result$final_annotations),
      stringsAsFactors = FALSE
    )
    
    # Create a bar plot
    p <- ggplot(plot_data, aes(x = Cluster, y = 1, fill = CellType)) +
      geom_bar(stat = "identity") +
      theme_minimal() +
      labs(title = "Cell Type Annotations", x = "Cluster", y = "") +
      theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
      coord_flip() +
      scale_fill_brewer(palette = "Set3")
    
    # Save the plot
    plot_file <- file.path(temp_log_dir, "annotations_plot.png")
    ggsave(plot_file, p, width = 10, height = 6)
    cat("\nCreated visualization at:", plot_file, "\n")
  }
}

cat("\nDone!\n")
