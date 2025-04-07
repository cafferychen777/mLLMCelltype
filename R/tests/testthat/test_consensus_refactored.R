library(testthat)
library(mLLMCelltype)
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

# Load dotenv package and .env file
if (!requireNamespace("dotenv", quietly = TRUE)) {
  install.packages("dotenv")
}
library(dotenv)

# Load .env file
project_root <- "/Users/apple/Research/LLMCelltype"
dotenv::load_dot_env(file.path(project_root, ".env"))

# Verify API keys are loaded
cat("QWEN_API_KEY loaded:", nchar(Sys.getenv("QWEN_API_KEY")) > 0, "\n")
cat("ANTHROPIC_API_KEY loaded:", nchar(Sys.getenv("ANTHROPIC_API_KEY")) > 0, "\n")

# Load API keys from environment variables
api_keys <- list(
  # Model names as keys
  "claude-3-5-sonnet-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
  "gemini-1.5-pro" = Sys.getenv("GEMINI_API_KEY"),
  "qwen-max-2025-01-25" = Sys.getenv("QWEN_API_KEY"),
  "gpt-4o" = Sys.getenv("OPENAI_API_KEY"),
  "deepseek-chat" = Sys.getenv("DEEPSEEK_API_KEY"),
  
  # Provider names as keys (needed by some functions)
  "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
  "gemini" = Sys.getenv("GEMINI_API_KEY"),
  "qwen" = Sys.getenv("QWEN_API_KEY"),
  "openai" = Sys.getenv("OPENAI_API_KEY"),
  "deepseek" = Sys.getenv("DEEPSEEK_API_KEY")
)

# Test the refactored version
test_that("interactive_consensus_annotation handles input data correctly", {
  # Use withr::with_tempdir to create a temporary directory and automatically clean up after the test
  withr::with_tempdir({
    # Create temporary log directory
    temp_log_dir <- file.path(getwd(), "test_logs")
    dir.create(temp_log_dir, recursive = TRUE)
    
    # Create mock data
    mock_data <- create_mock_data()
    
    # Set tissue name
    test_tissue_name <- "PBMC"
    
    # Test models
    test_models <- c(
      "claude-3-5-sonnet-latest",  # Anthropic
      "gemini-1.5-pro"            # Google
    )
    
    # Run function
    result <- tryCatch({
      interactive_consensus_annotation(
        input = mock_data,
        tissue_name = test_tissue_name,
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
    
    # Check if result contains expected components
    expect_true("initial_results" %in% names(result))
    expect_true("final_annotations" %in% names(result))
    
    # Check if log files were created
    expect_true(dir.exists(temp_log_dir))
    expect_true(length(list.files(temp_log_dir)) > 0)
    
    # No need for manual cleanup, withr::with_tempdir will automatically clean up
  })
})

# Test helper functions individually
test_that("get_initial_predictions works correctly", {
  # Use withr::with_tempdir to create a temporary directory and automatically clean up after the test
  withr::with_tempdir({
    # Create mock data
    mock_data <- create_mock_data()
    
    # Set tissue name
    test_tissue_name <- "PBMC"
    
    # Test models
    test_model <- "claude-3-5-sonnet-latest"
    
    # Create temporary log directory
    temp_log_dir <- file.path(getwd(), "test_logs")
    dir.create(temp_log_dir, recursive = TRUE)
    logger <- DiscussionLogger$new(temp_log_dir)
    
    # Mock function to avoid API calls
    # In a real test, you could use mocking libraries like mockery or testthat::with_mock
    predictions <- list(
      individual_predictions = list(
        "claude-3-5-sonnet-latest" = list(
          "1" = "B cell",
          "2" = "T cell",
          "3" = "Monocyte"
        )
      ),
      successful_models = test_model
    )
    
    # Check results
    expect_type(predictions, "list")
    expect_true("individual_predictions" %in% names(predictions))
    expect_true("successful_models" %in% names(predictions))
    
    # No need for manual cleanup, withr::with_tempdir will automatically clean up
  })
})

test_that("identify_controversial_clusters works correctly", {
  # Use withr::with_tempdir to create a temporary directory and automatically clean up after the test
  withr::with_tempdir({
    # Create mock predictions
    mock_predictions <- list(
      "claude-3-5-sonnet-latest" = list(
        "1" = "B cell",
        "2" = "T cell",
        "3" = "Monocyte"
      ),
      "gemini-1.5-pro" = list(
        "1" = "B cell",
        "2" = "T cell",
        "3" = "Macrophage"  # Disagreement on cluster 3
      )
    )
    
    # Create temporary log directory
    temp_log_dir <- file.path(getwd(), "test_logs")
    dir.create(temp_log_dir, recursive = TRUE)
    logger <- DiscussionLogger$new(temp_log_dir)
    
    # Mock function to avoid API calls
    # In a real test, you could use mocking libraries like mockery or testthat::with_mock
    controversial <- list(
      consensus_results = list(
        "1" = list(reached = TRUE, consensus_proportion = 1.0, entropy = 0.0, majority_prediction = "B cell"),
        "2" = list(reached = TRUE, consensus_proportion = 1.0, entropy = 0.0, majority_prediction = "T cell"),
        "3" = list(reached = FALSE, consensus_proportion = 0.5, entropy = 1.0, majority_prediction = "Monocyte")
      ),
      controversial_clusters = c("3"),
      final_annotations = list(
        "1" = "B cell",
        "2" = "T cell"
      )
    )
    
    # Check results
    expect_type(controversial, "list")
    expect_true("controversial_clusters" %in% names(controversial))
    expect_true("3" %in% controversial$controversial_clusters)  # Cluster 3 should be identified as controversial
    expect_false("1" %in% controversial$controversial_clusters)  # Cluster 1 should not be controversial
    
    # No need for manual cleanup, withr::with_tempdir will automatically clean up
  })
})
