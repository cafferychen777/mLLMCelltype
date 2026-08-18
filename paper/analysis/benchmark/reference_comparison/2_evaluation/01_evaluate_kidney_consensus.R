# Evaluation script for kidney cell type benchmark
library(LLMCellType)

# Set paths
project_root <- "."

# Read and parse .env file
env_lines <- readLines(file.path(project_root, ".env"))
for(line in env_lines) {
  if(grepl("^[[:alpha:]].*=.*$", line)) {
    parts <- strsplit(line, "=")[[1]]
    key <- trimws(parts[1])
    value <- trimws(gsub('"', '', parts[2]))
    do.call(Sys.setenv, structure(list(value), names = key))
  }
}

# Create results directories if they don't exist
dir.create(file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/logs"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/cache"), recursive = TRUE, showWarnings = FALSE)

# Load preprocessed data
markers_list <- readRDS(file.path(project_root, "results/benchmark/reference_comparison/1_preprocessing/kidney_markers.rds"))

# Print data structure for verification
print("Loaded data structure:")
str(markers_list)

# Print API key status
print("\nAPI Key Status:")
print(paste("ANTHROPIC_API_KEY:", if(nchar(Sys.getenv("ANTHROPIC_API_KEY")) > 0) "Set" else "Not Set"))
print(paste("GEMINI_API_KEY:", if(nchar(Sys.getenv("GEMINI_API_KEY")) > 0) "Set" else "Not Set"))
print(paste("QWEN_API_KEY:", if(nchar(Sys.getenv("QWEN_API_KEY")) > 0) "Set" else "Not Set"))
print(paste("OPENAI_API_KEY:", if(nchar(Sys.getenv("OPENAI_API_KEY")) > 0) "Set" else "Not Set"))

# Run consensus annotation
results <- interactive_consensus_annotation(
  input = markers_list,
  tissuename = "kidney",
  models = c("claude-3-5-sonnet-latest",
             "claude-3-5-haiku-latest",
             "gemini-1.5-pro",
             "gemini-2.0-flash-exp",
             "qwen-max-2025-01-25",
             "gpt-4o"),
  api_keys = list(
    "claude-3-5-sonnet-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
    "claude-3-5-haiku-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
    "gemini-1.5-pro" = Sys.getenv("GEMINI_API_KEY"),
    "gemini-2.0-flash-exp" = Sys.getenv("GEMINI_API_KEY"),
    "qwen-max-2025-01-25" = Sys.getenv("QWEN_API_KEY"),
    "gpt-4o" = Sys.getenv("OPENAI_API_KEY")
  ),
  log_dir = file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/logs"),
  cache_dir = file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/cache")
)

# Save results
saveRDS(results, file.path(project_root, "results/benchmark/reference_comparison/2_evaluation/kidney_results.rds"))

print("Evaluation complete. Results saved to kidney_results.rds")
