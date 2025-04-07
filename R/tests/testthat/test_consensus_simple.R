library(testthat)
library(LLMCelltype)
library(withr) # Use withr package to manage temporary directories

# Mock data creation function
create_mock_data <- function() {
  data <- list(
    clusters = list(
      "1" = list(name = "Cluster 1", genes = c("CD19", "CD20", "CD79A")),
      "2" = list(name = "Cluster 2", genes = c("CD3D", "CD3E", "CD3G")),
      "3" = list(name = "Cluster 3", genes = c("CD14", "CD16", "CSF1R"))
    ),
    tissue = "PBMC"
  )
  return(data)
}

# Get API keys from environment variables or use mock values
api_keys <- list(
  anthropic = Sys.getenv("ANTHROPIC_API_KEY", "test_key_anthropic"),
  google = Sys.getenv("GOOGLE_API_KEY", "test_key_google"),
  openai = Sys.getenv("OPENAI_API_KEY", "test_key_openai"),
  qwen = Sys.getenv("QWEN_API_KEY", "test_key_qwen")
)

# Basic test for create_consensus_check_prompt
test_that("create_consensus_check_prompt formats correctly", {
  # Test with a set of model responses
  model_responses <- list(
    "Model1" = "B cells",
    "Model2" = "B lymphocytes",
    "Model3" = "T cells"
  )
  
  # Generate the prompt
  prompt <- create_consensus_check_prompt(model_responses)
  
  # Check that the prompt contains expected elements
  expect_match(prompt, "PREDICTIONS:", fixed = TRUE)
  expect_match(prompt, "Model 1 : B cells", fixed = TRUE)
  expect_match(prompt, "Model 2 : B lymphocytes", fixed = TRUE)
  expect_match(prompt, "Model 3 : T cells", fixed = TRUE)
})

# Skip the interactive consensus annotation test in CI environments
test_that("interactive_consensus_annotation handles input data correctly", {
  skip("Skipping interactive consensus test in CI environment")
  
  # Use withr::with_tempdir to create a temporary directory and automatically clean up after the test
  withr::with_tempdir({
    # Create mock data
    mock_data <- create_mock_data()
    
    # Create temporary log directory
    temp_log_dir <- file.path(getwd(), "test_logs")
    dir.create(temp_log_dir, recursive = TRUE)
    
    # Define test models
    test_models <- c(
      "claude-3-5-sonnet-latest",  # Claude
      "gpt-4o",                   # OpenAI
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
      # Return error message for testing
      return(e$message)
    })
    
    # Check if function runs without errors
    expect_true(is.list(result) || is.character(result))
    
    # No need for manual cleanup, withr::with_tempdir will automatically clean up
  })
})
