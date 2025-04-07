# Test script to check environment variables

# Print environment variables
cat("QWEN_API_KEY:", Sys.getenv("QWEN_API_KEY"), "\n")
cat("ANTHROPIC_API_KEY:", Sys.getenv("ANTHROPIC_API_KEY"), "\n")
cat("OPENAI_API_KEY:", Sys.getenv("OPENAI_API_KEY"), "\n")

# Try to load .env file
cat("\nAttempting to load .env file...\n")
tryCatch({
  # Check if dotenv package is available
  if (!requireNamespace("dotenv", quietly = TRUE)) {
    cat("dotenv package not installed. Installing...\n")
    install.packages("dotenv")
  }
  
  # Load dotenv package
  library(dotenv)
  
  # Load .env file
  dotenv::load_dot_env("/Users/apple/Research/LLMCelltype/.env")
  
  cat(".env file loaded successfully!\n")
}, error = function(e) {
  cat("Error loading .env file:", e$message, "\n")
})

# Print environment variables again
cat("\nAfter loading .env file:\n")
cat("QWEN_API_KEY:", Sys.getenv("QWEN_API_KEY"), "\n")
cat("ANTHROPIC_API_KEY:", Sys.getenv("ANTHROPIC_API_KEY"), "\n")
cat("OPENAI_API_KEY:", Sys.getenv("OPENAI_API_KEY"), "\n")
