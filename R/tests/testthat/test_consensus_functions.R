library(testthat)
library(LLMCelltype)
library(withr) # Use withr package to manage temporary directories

# Mock data creation function
create_mock_data <- function() {
  # Create a list of genes for each cluster
  data <- list(
    "1" = list(
      genes = c("CD19", "CD20", "CD79A", "CD79B", "PAX5"),
      description = "B cell markers"
    ),
    "2" = list(
      genes = c("CD3D", "CD3E", "CD4", "CD8A", "IL7R"),
      description = "T cell markers"
    ),
    "3" = list(
      genes = c("CD14", "CD68", "CD163", "CSF1R", "ITGAM"),
      description = "Monocyte markers"
    )
  )
  return(data)
}

# Read and parse .env file
project_root <- "/Users/apple/Research/LLMCelltype"
env_lines <- readLines(file.path(project_root, ".env"))
for(line in env_lines) {
  if(grepl("^[[:alpha:]].*=.*$", line)) {
    parts <- strsplit(line, "=")[[1]]
    key <- trimws(parts[1])
    value <- trimws(gsub('"', '', parts[2]))
    do.call(Sys.setenv, structure(list(value), names = key))
  }
}

# Load API keys from environment variables
api_keys <- list(
  "claude-3-5-sonnet-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
  "gemini-1.5-pro" = Sys.getenv("GEMINI_API_KEY"),
  "qwen-max-2025-01-25" = Sys.getenv("QWEN_API_KEY"),
  "gpt-4o" = Sys.getenv("OPENAI_API_KEY"),
  "deepseek-chat" = Sys.getenv("DEEPSEEK_API_KEY")
)

# Test cases
test_that("interactive_consensus_annotation handles input data correctly", {
  # Use withr::with_tempdir to create a temporary directory and automatically clean up after the test
  withr::with_tempdir({
    # Create temporary log directory
    temp_log_dir <- file.path(getwd(), "test_logs")
    dir.create(temp_log_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Create mock data
    mock_data <- create_mock_data()
    
    # Test models
    test_models <- c(
      "claude-3-5-sonnet-latest",  # Anthropic
      "gemini-1.5-pro",           # Google
      "qwen-max-2025-01-25"       # Qwen
    )
    
    # Run function
    result <- tryCatch({
      interactive_consensus_annotation(
        input = mock_data,
        tissue_name = "PBMC",
        models = test_models,
        api_keys = api_keys,
        top_gene_count = 5,
        controversy_threshold = 0.7,
        log_dir = temp_log_dir
      )
    }, error = function(e) {
      message("Error occurred during annotation: ", e$message)
      e
    })
    
    # Basic structure test
    expect_false(inherits(result, "error"))
    expect_type(result, "list")
    
    # Check if log files were created
    expect_true(dir.exists(temp_log_dir))
    expect_true(length(list.files(temp_log_dir)) > 0)
    
    # No need for manual cleanup, withr::with_tempdir will automatically clean up
  })
})



test_that("DiscussionLogger works correctly", {
  # Use withr::with_tempdir to create a temporary directory and automatically clean up after the test
  withr::with_tempdir({
    # Create temporary log directory
    temp_log_dir <- file.path(getwd(), "test_logs")
    dir.create(temp_log_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Initialize logger
    logger <- DiscussionLogger$new(temp_log_dir)
    
    # Test logging
    cluster_id <- 1
    tissue_name <- "PBMC"
    marker_genes <- c("CD19", "CD20", "CD79A")
    
    # Start cluster discussion
    logger$start_cluster_discussion(cluster_id, tissue_name, marker_genes)
    
    # Record events
    logger$log_prediction("claude", 1, list(cell_type = "B cell", confidence = 0.9))
    logger$log_consensus_check(1, TRUE, 0.9)
    logger$log_final_consensus("B cell", "High confidence B cell annotation")
    
    # End discussion
    logger$end_cluster_discussion()
    
    # Check if log files exist
    log_files <- list.files(temp_log_dir, recursive = TRUE)
    expect_true(length(log_files) > 0)
    
    # No need for manual cleanup, withr::with_tempdir will automatically clean up
  })
})
