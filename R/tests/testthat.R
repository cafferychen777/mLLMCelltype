# This file is part of the standard setup for testthat.
# Modified to skip tests when no test files are present.

library(testthat)
library(mLLMCelltype)

# Check if test files exist before running tests
test_files <- list.files("testthat", pattern = "^test.*\\.R$", full.names = TRUE)
if (length(test_files) > 0) {
  test_check("mLLMCelltype")
} else {
  message("No test files found, skipping tests")
}
