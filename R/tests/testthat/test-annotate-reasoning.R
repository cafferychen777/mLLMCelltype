# Tests for annotate_cell_types return_reasoning functionality

test_that("annotation prompt variants share normalized prompt data", {
  inputs <- list(
    list("10" = c(" G10 ", "G10b"), "2" = list(genes = "G2")),
    data.frame(
      cluster = c(" 10 ", "2", "10"),
      gene = c(" G10a ", "G2", "G10b"),
      avg_log2FC = c(2, 3, 1)
    )
  )

  for (input in inputs) {
    standard <- create_annotation_prompt(input, " PBMC ", top_gene_count = 2)
    reasoning <- create_reasoning_annotation_prompt(
      input,
      " PBMC ",
      top_gene_count = 2
    )

    expect_identical(reasoning$expected_count, standard$expected_count)
    expect_identical(reasoning$gene_lists, standard$gene_lists)
    expect_identical(names(reasoning$gene_lists), c("2", "10"))

    marker_lines <- paste0(
      names(standard$gene_lists),
      ": ",
      unlist(standard$gene_lists, use.names = FALSE)
    )
    for (marker_line in marker_lines) {
      expect_match(standard$prompt, marker_line, fixed = TRUE)
      expect_match(reasoning$prompt, marker_line, fixed = TRUE)
    }
  }
})

test_that("annotate_cell_types return_reasoning=TRUE parses JSON array response", {
  json_response <- paste0(
    '[{"cluster_id": "0", "cell_type": "T cells", ',
    '"marker_genes": "CD3D, CD3E", ',
    '"gene_expression": "CD3D is highly expressed in T cells"}, ',
    '{"cluster_id": "1", "cell_type": "B cells", ',
    '"marker_genes": "CD19, CD79A", ',
    '"gene_expression": "CD19 is highly expressed in B cells"}]'
  )

  result <- testthat::with_mocked_bindings({
    annotate_cell_types(
      input = list(
        "0" = list(genes = c("CD3D", "CD3E")),
        "1" = list(genes = c("CD19", "CD79A"))
      ),
      tissue_name = "PBMC",
      model = "gpt-5.5",
      api_key = "test-key",
      return_reasoning = TRUE
    )
  },
  get_model_response = function(prompt, model, api_key, base_urls, normalize = TRUE) json_response,
  log_info = function(...) NULL,
  log_debug = function(...) NULL)

  expect_type(result, "list")
  expect_named(result, c("0", "1"))
  expect_identical(result[["0"]]$cell_type, "T cells")
  expect_identical(result[["0"]]$marker_genes, "CD3D, CD3E")
  expect_match(result[["0"]]$gene_expression, "CD3D")
  expect_identical(result[["1"]]$cell_type, "B cells")
})


test_that("annotate_cell_types return_reasoning=TRUE parses JSON object response", {
  json_response <- paste0(
    '{"0": {"cell_type": "T cells", "marker_genes": "CD3D", ',
    '"gene_expression": "CD3D in T cells"}, ',
    '"1": {"cell_type": "B cells", "marker_genes": "CD19", ',
    '"gene_expression": "CD19 in B cells"}}'
  )

  result <- testthat::with_mocked_bindings({
    annotate_cell_types(
      input = list(
        "0" = list(genes = c("CD3D")),
        "1" = list(genes = c("CD19"))
      ),
      tissue_name = "PBMC",
      model = "gpt-5.5",
      api_key = "test-key",
      return_reasoning = TRUE
    )
  },
  get_model_response = function(prompt, model, api_key, base_urls, normalize = TRUE) json_response,
  log_info = function(...) NULL,
  log_debug = function(...) NULL)

  expect_identical(result[["0"]]$cell_type, "T cells")
  expect_identical(result[["1"]]$cell_type, "B cells")
})


test_that("annotate_cell_types return_reasoning preserves marker gene case", {
  json_response <- '[{"cluster_id": "0", "cell_type": "T cells", "marker_genes": "Cd3d, CD3E", "gene_expression": "..."}]'

  result <- testthat::with_mocked_bindings({
    annotate_cell_types(
      input = list("0" = list(genes = c("Cd3d", "CD3E"))),
      tissue_name = "PBMC",
      model = "gpt-5.5",
      api_key = "test-key",
      return_reasoning = TRUE
    )
  },
  get_model_response = function(prompt, model, api_key, base_urls, normalize = TRUE) json_response,
  log_info = function(...) NULL,
  log_debug = function(...) NULL)

  expect_identical(result[["0"]]$marker_genes, "Cd3d, CD3E")
})


test_that("annotate_cell_types return_reasoning falls back to Unknown on invalid JSON", {
  result <- testthat::with_mocked_bindings({
    annotate_cell_types(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      model = "gpt-5.5",
      api_key = "test-key",
      return_reasoning = TRUE
    )
  },
  get_model_response = function(prompt, model, api_key, base_urls, normalize = TRUE) "not valid json",
  log_info = function(...) NULL,
  log_debug = function(...) NULL)

  expect_identical(result[["0"]]$cell_type, "Unknown")
  expect_identical(result[["0"]]$marker_genes, "")
  expect_identical(result[["0"]]$gene_expression, "")
})


test_that("annotate_cell_types return_reasoning=FALSE behavior is unchanged", {
  result <- testthat::with_mocked_bindings({
    annotate_cell_types(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      model = "gpt-5.5",
      api_key = "test-key",
      return_reasoning = FALSE
    )
  },
  get_model_response = function(prompt, model, api_key, base_urls, normalize = TRUE) c("T cells"),
  log_info = function(...) NULL,
  log_debug = function(...) NULL)

  expect_identical(result, c("T cells"))
})


test_that("annotate_cell_types return_reasoning=TRUE parses markdown-fenced multi-line JSON", {
  json_response <- paste(
    "```json",
    '[',
    '  {',
    '    "cluster_id": "0",',
    '    "cell_type": "T cells",',
    '    "marker_genes": "CD3D, CD3E",',
    '    "gene_expression": "CD3D and CD3E are T cell markers"',
    '  },',
    '  {',
    '    "cluster_id": "1",',
    '    "cell_type": "B cells",',
    '    "marker_genes": "CD19, CD79A",',
    '    "gene_expression": "CD19 and CD79A are B cell markers"',
    '  }',
    ']',
    "```",
    sep = "\n"
  )

  result <- testthat::with_mocked_bindings({
    annotate_cell_types(
      input = list(
        "0" = list(genes = c("CD3D", "CD3E")),
        "1" = list(genes = c("CD19", "CD79A"))
      ),
      tissue_name = "PBMC",
      model = "kimi-k2.6",
      api_key = "test-key",
      return_reasoning = TRUE
    )
  },
  get_model_response = function(prompt, model, api_key, base_urls, normalize = TRUE) json_response,
  log_info = function(...) NULL,
  log_debug = function(...) NULL)

  expect_identical(result[["0"]]$cell_type, "T cells")
  expect_identical(result[["0"]]$marker_genes, "CD3D, CD3E")
  expect_match(result[["0"]]$gene_expression, "CD3D")
  expect_identical(result[["1"]]$cell_type, "B cells")
  expect_identical(result[["1"]]$marker_genes, "CD19, CD79A")
})


test_that("annotate_cell_types return_reasoning=TRUE with api_key=NA returns prompt", {
  prompt <- annotate_cell_types(
    input = list("0" = list(genes = c("CD3D"))),
    tissue_name = "PBMC",
    model = "gpt-5.5",
    api_key = NA,
    return_reasoning = TRUE
  )

  expect_type(prompt, "character")
  expect_match(prompt, "0: CD3D", fixed = TRUE)
  expect_match(prompt, "CD3D", fixed = TRUE)
  expect_match(prompt, "JSON object", fixed = TRUE)
  expect_match(prompt, "marker_genes", fixed = TRUE)
  expect_match(prompt, "gene_expression", fixed = TRUE)
})
