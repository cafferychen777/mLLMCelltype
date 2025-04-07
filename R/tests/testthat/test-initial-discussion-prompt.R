library(testthat)
library(mLLMCelltype)

test_that("create_initial_discussion_prompt formats correctly", {
  # Test data
  cluster_id <- 1
  cluster_genes <- "CD19, CD20, CD79A, CD79B, PAX5"
  tissuename <- "PBMC"
  initial_predictions <- list(
    "Model1" = c("B cell"),
    "Model2" = c("B lymphocyte"),
    "Model3" = c("T cell")
  )
  
  # Generate the prompt
  prompt <- create_initial_discussion_prompt(
    cluster_id = cluster_id,
    cluster_genes = cluster_genes,
    tissue_name = tissuename,
    initial_predictions = initial_predictions
  )
  
  # Check that the prompt is a string
  expect_type(prompt, "character")
  
  # Check that the prompt contains the cluster information
  expect_true(grepl(sprintf("cluster %d", cluster_id), prompt))
  expect_true(grepl(cluster_genes, prompt, fixed = TRUE))
  expect_true(grepl(tissuename, prompt, fixed = TRUE))
  
  # Check that the prompt contains the model predictions
  expect_true(grepl("Model1: B cell", prompt, fixed = TRUE))
  expect_true(grepl("Model2: B lymphocyte", prompt, fixed = TRUE))
  expect_true(grepl("Model3: T cell", prompt, fixed = TRUE))
  
  # Check that the prompt contains the Toulmin model structure
  expect_true(grepl("CLAIM:", prompt, fixed = TRUE))
  expect_true(grepl("GROUNDS/DATA:", prompt, fixed = TRUE))
  expect_true(grepl("WARRANT:", prompt, fixed = TRUE))
  expect_true(grepl("BACKING:", prompt, fixed = TRUE))
  expect_true(grepl("QUALIFIER:", prompt, fixed = TRUE))
  expect_true(grepl("REBUTTAL:", prompt, fixed = TRUE))
  
  # Check that the prompt contains the response format
  expect_true(grepl("CELL TYPE:", prompt, fixed = TRUE))
  expect_true(grepl("GROUNDS:", prompt, fixed = TRUE))
  expect_true(grepl("WARRANT:", prompt, fixed = TRUE))
  expect_true(grepl("BACKING:", prompt, fixed = TRUE))
  expect_true(grepl("QUALIFIER:", prompt, fixed = TRUE))
  expect_true(grepl("REBUTTAL:", prompt, fixed = TRUE))
})
