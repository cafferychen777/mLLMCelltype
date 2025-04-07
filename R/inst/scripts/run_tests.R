# Script to run tests for the refactored consensus annotation code

# Load required packages
library(testthat)
library(devtools)

# Set working directory to the package root
setwd("/Users/apple/Research/LLMCelltype/code/R")

# Load the package
load_all(".")

# Run the tests
test_file("tests/testthat/To https://github.com/cafferychen777/LLMCelltype.git
   3556a5a..1d2fd80  main -> main
.R")

# Print completion message
cat("\nTests completed!\n")
