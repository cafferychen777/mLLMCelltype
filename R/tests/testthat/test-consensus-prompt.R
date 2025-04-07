library(testthat)
library(LLMCelltype)

test_that("create_consensus_check_prompt formats responses correctly", {
  # Test with a set of model responses
  responses <- c(
    "B cell",
    "B lymphocyte",
    "T cell"
  )
  
  # Generate the prompt
  prompt <- create_consensus_check_prompt(responses)
  
  # Check that the prompt is a string
  expect_type(prompt, "character")
  
  # Check that the prompt contains the model predictions
  expect_true(grepl("Model 1 : B cell", prompt, fixed = TRUE))
  expect_true(grepl("Model 2 : B lymphocyte", prompt, fixed = TRUE))
  expect_true(grepl("Model 3 : T cell", prompt, fixed = TRUE))
  
  # Check that the prompt contains the important sections
  expect_true(grepl("PREDICTIONS:", prompt, fixed = TRUE))
  expect_true(grepl("IMPORTANT GUIDELINES:", prompt, fixed = TRUE))
  expect_true(grepl("CALCULATE THE FOLLOWING METRICS:", prompt, fixed = TRUE))
  expect_true(grepl("RESPONSE FORMAT:", prompt, fixed = TRUE))
  
  # Check that the prompt contains the example matches
  expect_true(grepl("'NK cells' = 'Natural Killer cells'", prompt, fixed = TRUE))
  expect_true(grepl("'CD8+ T cells' = 'Cytotoxic T cells'", prompt, fixed = TRUE))
  expect_true(grepl("'B cells' = 'B lymphocytes'", prompt, fixed = TRUE))
})
