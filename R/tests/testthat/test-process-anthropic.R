library(testthat)
library(mockery)
library(httr)
library(jsonlite)
library(devtools)

# Load the package
load_all(".")

test_that("process_anthropic handles basic request correctly", {
  skip_if_not_installed("mockery")
  skip_if_not_installed("httr")
  
  # Mock the API response
  mock_response <- structure(
    list(
      url = "https://api.anthropic.com/v1/messages",
      status_code = 200,
      headers = list(`Content-Type` = "application/json"),
      content = charToRaw(jsonlite::toJSON(list(
        content = list(
          list(text = "This is a test response\nWith multiple lines")
        )
      ), auto_unbox = TRUE))
    ),
    class = c("response", "list")
  )
  
  # Create mock for httr::POST
  mockery::stub(
    process_anthropic,
    "httr::POST",
    mock_response
  )
  
  # Create mock for write_log to avoid console output during tests
  mockery::stub(
    process_anthropic,
    "write_log",
    function(...) invisible(NULL)
  )
  
  # Test input
  prompt <- "Test prompt\nWith multiple lines"
  model <- "claude-3-sonnet-20240229"
  api_key <- "test_key"
  
  # Run the function
  result <- process_anthropic(prompt, model, api_key)
  
  # Verify the result
  expect_type(result, "character")
  expect_length(result, 2)
  expect_equal(result[1], "This is a test response")
  expect_equal(result[2], "With multiple lines")
})



test_that("process_anthropic works in check_consensus scenario", {
  skip_if_not_installed("mockery")
  skip_if_not_installed("httr")
  
  # Mock response for consensus check
  mock_consensus_response <- structure(
    list(
      url = "https://api.anthropic.com/v1/messages",
      status_code = 200,
      headers = list(`Content-Type` = "application/json"),
      content = charToRaw(jsonlite::toJSON(list(
        content = list(
          list(text = "1\n0.8")  # First line: consensus reached (1), Second line: confidence score (0.8)
        )
      ), auto_unbox = TRUE))
    ),
    class = c("response", "list")
  )
  
  # Setup mock
  stub(process_anthropic, "httr::POST", mock_consensus_response)
  
  # Test input similar to check_consensus function
  prompt <- "Please analyze these model responses and determine if there is consensus:\nModel 1: T cell\nModel 2: T lymphocyte\nModel 3: T cell"
  result <- process_anthropic(prompt, "claude-3-5-sonnet-latest", "test_key")
  
  # Verify results
  expect_equal(length(result), 2)  # Should return two lines
  expect_equal(result[1], "1")   # Consensus indicator
  expect_equal(result[2], "0.8") # Confidence score
})

test_that("process_anthropic works in batch annotation scenario", {
  skip_if_not_installed("mockery")
  skip_if_not_installed("httr")
  
  # Mock response for batch annotation
  mock_batch_response <- structure(
    list(
      url = "https://api.anthropic.com/v1/messages",
      status_code = 200,
      headers = list(`Content-Type` = "application/json"),
      content = charToRaw(jsonlite::toJSON(list(
        content = list(
          list(text = "1. CD4+ T cells\n2. CD8+ T cells\n3. B cells\n4. NK cells\n5. Monocytes")
        )
      ), auto_unbox = TRUE))
    ),
    class = c("response", "list")
  )
  
  # Setup mock
  stub(process_anthropic, "httr::POST", mock_batch_response)
  
  # Test input similar to batch annotation in annotate_cell_types
  prompt <- "You are a cell type annotation expert. Below are marker genes for different cell clusters in human PBMC.\n\n1: CD4, CD3D, IL7R, CCR7\n2: CD8A, CD3D, NKG7\n3: CD19, CD79A, MS4A1\n4: GNLY, NKG7, GZMB\n5: CD14, LYZ, CSF1R\n\nFor each numbered cluster, provide only the cell type name in a new line, without any explanation."
  result <- process_anthropic(prompt, "claude-3-5-sonnet-latest", "test_key")
  
  # Verify results
  expect_length(result, 5)  # Should return five lines
  expect_match(result[1], "CD4\\+ T cells")
  expect_match(result[2], "CD8\\+ T cells")
  expect_match(result[3], "B cells")
  expect_match(result[4], "NK cells")
  expect_match(result[5], "Monocytes")
})

test_that("process_anthropic works in discussion summary scenario", {
  skip_if_not_installed("mockery")
  skip_if_not_installed("httr")
  
  # Mock response for discussion summary
  mock_summary_response <- structure(
    list(
      url = "https://api.anthropic.com/v1/messages",
      status_code = 200,
      headers = list(`Content-Type` = "application/json"),
      content = charToRaw(jsonlite::toJSON(list(
        content = list(
          list(text = "Summary of Discussion:\n\nCluster 3 has been identified as CD14+ Monocytes based on the following key points:\n\n1. High expression of monocyte markers CD14, CD16, and CSF1R\n2. Strong evidence of myeloid lineage\n3. Consistent with tissue context (PBMC)\n4. Agreement among 4 out of 5 models\n\nConfidence: High (90%)\nFinal Annotation: CD14+ Monocytes")
        )
      ), auto_unbox = TRUE))
    ),
    class = c("response", "list")
  )
  
  # Setup mock
  stub(process_anthropic, "httr::POST", mock_summary_response)
  
  # Test input similar to discussion summary
  prompt <- "Please summarize the discussion about Cluster 3 and provide a final annotation.\n\nModel 1: CD14+ Monocytes\nModel 2: Classical Monocytes\nModel 3: CD14+ Monocytes\nModel 4: CD14+ Monocytes\nModel 5: Macrophages\n\nKey markers: CD14, CD16, CSF1R, LYZ, FCGR3A\nTissue: PBMC\n\nProvide a structured summary including:\n1. Key points from the discussion\n2. Confidence level\n3. Final annotation"
  result <- process_anthropic(prompt, "claude-3-5-sonnet-latest", "test_key")
  
  # Verify results
  expect_true(length(result) > 5)  # Should return multiple lines
  expect_match(result[1], "Summary of Discussion")
  expect_match(paste(result, collapse = "\n"), "CD14\\+ Monocytes")
  expect_match(paste(result, collapse = "\n"), "Confidence: High")
  expect_match(paste(result, collapse = "\n"), "Final Annotation")
})

test_that("process_anthropic works in cell type annotation scenario", {
  skip_if_not_installed("mockery")
  skip_if_not_installed("httr")
  
  # Mock response for cell type annotation
  mock_annotation_response <- structure(
    list(
      url = "https://api.anthropic.com/v1/messages",
      status_code = 200,
      headers = list(`Content-Type` = "application/json"),
      content = charToRaw(jsonlite::toJSON(list(
        content = list(
          list(text = "Cell Type: CD4+ T cells\nConfidence: High\nRationale: High expression of CD4 and CD3D markers, which are characteristic of CD4+ T cells.")
        )
      ), auto_unbox = TRUE))
    ),
    class = c("response", "list")
  )
  
  # Setup mock
  stub(process_anthropic, "httr::POST", mock_annotation_response)
  
  # Test input similar to annotate_cell_types function
  prompt <- "Please analyze these marker genes and suggest a cell type:\nTop markers: CD4, CD3D, IL7R, CCR7\nTissue: PBMC"
  result <- process_anthropic(prompt, "claude-3-5-sonnet-latest", "test_key")
  
  # Verify results
  expect_length(result, 3)  # Should return three lines
  expect_match(result[1], "Cell Type: CD4\\+ T cells")
  expect_match(result[2], "Confidence: High")
  expect_match(result[3], "Rationale:")
})



