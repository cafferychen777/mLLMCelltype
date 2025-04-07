# Script to test the modified check_consensus function

# Load required packages
library(devtools)

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

# Create a function to test the check_consensus function
test_check_consensus <- function() {
  # Create test data
  test_responses <- c("T cell", "T cell", "B cell")
  
  # Create API keys with both model names and provider names
  api_keys <- list(
    # Model names as keys
    "claude-3-5-sonnet-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
    "gemini-1.5-pro" = Sys.getenv("GEMINI_API_KEY"),
    "qwen-max-2025-01-25" = Sys.getenv("QWEN_API_KEY"),
    "gpt-4o" = Sys.getenv("OPENAI_API_KEY"),
    "deepseek-chat" = Sys.getenv("DEEPSEEK_API_KEY"),
    
    # Provider names as keys
    "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
    "gemini" = Sys.getenv("GEMINI_API_KEY"),
    "qwen" = Sys.getenv("QWEN_API_KEY"),
    "openai" = Sys.getenv("OPENAI_API_KEY"),
    "deepseek" = Sys.getenv("DEEPSEEK_API_KEY")
  )
  
  # Load the package
  load_all(".")
  
  # Test check_consensus
  cat("\nTesting check_consensus with API keys...\n")
  tryCatch({
    result <- check_consensus(test_responses, api_keys)
    cat("check_consensus result:\n")
    print(result)
    return(TRUE)
  }, error = function(e) {
    cat("Error in check_consensus:", e$message, "\n")
    return(FALSE)
  })
}

# Run the test
test_result <- test_check_consensus()

cat("\nTest result:", if(test_result) "SUCCESS" else "FAILURE", "\n")
