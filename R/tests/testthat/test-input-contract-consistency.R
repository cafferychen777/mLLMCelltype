# Input contract consistency tests

make_test_json_response <- function(payload,
                                    status_code = 200L,
                                    url = "https://example.test/v1/chat/completions") {
  structure(
    list(
      url = url,
      status_code = status_code,
      headers = list("Content-Type" = "application/json"),
      all_headers = list(list(
        status = status_code,
        version = "HTTP/1.1",
        headers = list("Content-Type" = "application/json")
      )),
      cookies = data.frame(),
      content = charToRaw(jsonlite::toJSON(payload, auto_unbox = TRUE)),
      date = Sys.time(),
      times = c(),
      request = list(method = "POST")
    ),
    class = "response"
  )
}

enable_logger_file_writes <- function() {
  variable <- "_R_CHECK_PACKAGE_NAME_"
  previous <- Sys.getenv(variable, unset = NA_character_)
  Sys.unsetenv(variable)

  function() {
    if (is.na(previous)) {
      Sys.unsetenv(variable)
    } else {
      do.call(Sys.setenv, setNames(list(previous), variable))
    }
  }
}

preserve_environment_variables <- function(variables) {
  previous <- Sys.getenv(variables, unset = NA_character_)

  function() {
    Sys.unsetenv(variables)
    configured <- !is.na(previous)
    if (any(configured)) {
      do.call(
        Sys.setenv,
        setNames(as.list(previous[configured]), variables[configured])
      )
    }
  }
}

make_mock_builtin_processor_factory <- function(process_request) {
  force(process_request)
  function(provider, base_url = NULL) {
    list(process_request = process_request)
  }
}

test_that("normalize_cluster_gene_list canonicalizes list inputs", {
  unnamed <- normalize_cluster_gene_list(list(c("G1"), c("G2")))
  expect_identical(names(unnamed), c("0", "1"))

  numeric_named <- normalize_cluster_gene_list(
    list("1" = c("A"), "2" = list(genes = c("B")))
  )
  expect_identical(names(numeric_named), c("1", "2"))

  non_numeric_named <- normalize_cluster_gene_list(
    list("t_cells" = c("CD3D"), "b_cells" = list(genes = c("MS4A1")))
  )
  expect_identical(names(non_numeric_named), c("t_cells", "b_cells"))
})

test_that("normalize_cluster_gene_list cleans genes and rejects ambiguous cluster IDs", {
  normalized <- normalize_cluster_gene_list(list(
    " cluster_a " = list(genes = c(" CD3D ", NA, "", "CD3D", "CD3E"))
  ))
  expect_identical(normalized, list(cluster_a = c("CD3D", "CD3E")))

  expect_error(normalize_cluster_gene_list(list()), "at least one cluster")
  expect_error(
    normalize_cluster_gene_list(setNames(list(c("G1")), "")),
    "Cluster IDs must be non-empty strings"
  )
  expect_error(
    normalize_cluster_gene_list(list("a" = list(genes = list("G1")))),
    "marker genes must be an atomic vector"
  )
  expect_error(
    normalize_cluster_gene_list(list("a" = c(NA, "   "))),
    "No genes found for cluster"
  )
})

test_that("create_annotation_prompt accepts mixed list element formats", {
  prompt_result <- create_annotation_prompt(
    input = list(
      gs1 = c("CD4", "CD3D"),
      gs2 = list(genes = c("CD14"))
    ),
    tissue_name = "PBMC"
  )

  expect_true(all(c("gs1", "gs2") %in% names(prompt_result$gene_lists)))
  expect_match(prompt_result$prompt, "gs1: CD4, CD3D")
  expect_match(prompt_result$prompt, "gs2: CD14")
})

test_that("create_annotation_prompt returns the exact numeric display order", {
  prompt_result <- create_annotation_prompt(
    list("10" = "G10", "2" = "G2", "1" = "G1"),
    "PBMC"
  )

  expect_identical(names(prompt_result$gene_lists), c("1", "2", "10"))
  prompt_lines <- strsplit(prompt_result$prompt, "\n", fixed = TRUE)[[1]]
  cluster_lines <- prompt_lines[grepl("^(1|2|10):", prompt_lines)]
  expect_identical(cluster_lines, c("1: G1", "2: G2", "10: G10"))
  expect_match(prompt_result$prompt, "in the same order", fixed = TRUE)

  aligned <- align_model_predictions(
    c("Type 1", "Type 2", "Type 10"),
    names(prompt_result$gene_lists)
  )
  expect_identical(
    aligned,
    c("1" = "Type 1", "2" = "Type 2", "10" = "Type 10")
  )
})

test_that("create_annotation_prompt validates exported inputs and data frame schema", {
  expect_error(
    create_annotation_prompt(list("0" = c("G1")), "   "),
    "tissue_name must be a non-empty character scalar"
  )
  expect_error(
    create_annotation_prompt(list("0" = c("G1")), "PBMC", top_gene_count = 1.5),
    "top_gene_count must be a positive integer"
  )
  expect_error(
    create_annotation_prompt(data.frame(cluster = "0", gene = "G1"), "PBMC"),
    "exactly one each"
  )
  expect_error(
    create_annotation_prompt(
      data.frame(cluster = "0", gene = "G1", avg_log2FC = "high"),
      "PBMC"
    ),
    "avg_log2FC must be numeric"
  )

  result <- create_annotation_prompt(
    data.frame(
      cluster = c(" 1 ", "1", "2"),
      gene = c(" G1 ", "G1", "G2"),
      avg_log2FC = c(2, 1, 3)
    ),
    " PBMC ",
    top_gene_count = 2
  )
  expect_identical(names(result$gene_lists), c("1", "2"))
  expect_identical(result$gene_lists[["1"]], "G1")
  expect_match(result$prompt, "in PBMC")
})

test_that("annotation, discussion, and cache marker selection stay aligned", {
  markers <- data.frame(
    cluster = c(" 1 ", "1", "1", NA),
    gene = c(" LOW ", "HIGH", "HIGH", "IGNORED"),
    avg_log2FC = c(-1, 3, 2, 10)
  )

  prompt_result <- suppressWarnings(
    create_annotation_prompt(markers, "PBMC", top_gene_count = 2)
  )
  discussion_genes <- extract_cluster_genes_for_discussion(markers, "1", 2)

  expect_identical(prompt_result$gene_lists[["1"]], "HIGH,LOW")
  expect_identical(discussion_genes, "HIGH,LOW")

  cache_manager <- CacheManager$new("temp")
  expect_false(identical(
    cache_manager$generate_key(markers, "gpt-5.5", "1", "PBMC", 1),
    cache_manager$generate_key(markers, "gpt-5.5", "1", "PBMC", 2)
  ))
})

test_that("CacheManager generate_key supports character-vector list clusters", {
  cache_manager <- CacheManager$new(cache_dir = "temp")
  input_list <- list(
    "0" = c("CD3D", "CD3E"),
    "1" = list(genes = c("MS4A1"))
  )

  key_0 <- cache_manager$generate_key(input_list, c("gpt-5.5"), "0")
  key_1 <- cache_manager$generate_key(input_list, c("gpt-5.5"), "1")

  expect_true(is.character(key_0))
  expect_gt(nchar(key_0), 10)
  expect_false(identical(key_0, key_1))
})

test_that("parse_text_predictions resolves cluster aliases without positional reassignment", {
  parsed <- parse_text_predictions(
    c("Cluster 2: B cell"),
    all_clusters = c("1", "2")
  )

  expect_null(parsed[["1"]])
  expect_identical(parsed[["2"]], "B cell")

  complete <- parse_text_predictions(
    c("Cluster_1: T cell", "2) B cell"),
    all_clusters = c("1", "2")
  )
  expect_identical(complete, list("1" = "T cell", "2" = "B cell"))
})

test_that("parse_text_predictions uses positions only for unlabeled responses", {
  parsed <- parse_text_predictions(
    c("T-cell", "B cell"),
    all_clusters = c("alpha", "beta")
  )

  expect_identical(parsed, list(alpha = "T-cell", beta = "B cell"))
})

test_that("parse_text_predictions preserves exact IDs over ambiguous aliases", {
  exact <- parse_text_predictions(
    "Cluster 1: Named cluster",
    all_clusters = c("1", "Cluster 1")
  )
  expect_identical(exact, list("Cluster 1" = "Named cluster"))

  ambiguous <- parse_text_predictions(
    "Cluster_1: Ambiguous",
    all_clusters = c("1", "Cluster 1")
  )
  expect_length(ambiguous, 0)
})

test_that("parse_text_predictions maps a numbered list onto 0-based clusters without shifting", {
  # Regression: models commonly number their answers ("1.", "2.", ...). On
  # Seurat 0-based clusters the ordinals are list indices, not cluster IDs;
  # trusting them shifts every annotation one cluster and drops cluster 0.
  parsed <- parse_text_predictions(
    c("1. T cells", "2. B cells", "3. NK cells"),
    all_clusters = c("0", "1", "2")
  )
  expect_identical(parsed, list("0" = "T cells", "1" = "B cells", "2" = "NK cells"))
})

test_that("parse_text_predictions keeps a positional response despite a preamble header", {
  parsed <- parse_text_predictions(
    c("Here are the cell type annotations:", "T cells", "B cells", "NK cells"),
    all_clusters = c("0", "1", "2")
  )
  expect_identical(parsed, list("0" = "T cells", "1" = "B cells", "2" = "NK cells"))
})

test_that("parse_text_predictions keeps an annotation-internal colon whole", {
  parsed <- parse_text_predictions(
    c("Astrocytes", "Neurons: excitatory", "Microglia"),
    all_clusters = c("0", "1", "2")
  )
  expect_identical(
    parsed,
    list("0" = "Astrocytes", "1" = "Neurons: excitatory", "2" = "Microglia")
  )
})

test_that("parse_text_predictions keeps a mid-list Unknown in its own slot", {
  # Regression: dropping a sentinel line must not shift later clusters up one.
  parsed <- parse_text_predictions(
    c("T cells", "Unknown", "B cells"),
    all_clusters = c("0", "1", "2")
  )
  # cluster 1 is left unassigned (-> Unknown downstream); cluster 2 keeps B cells.
  expect_identical(parsed, list("0" = "T cells", "2" = "B cells"))
})

test_that("parse_text_predictions ignores a stray Summary: line and keeps labels", {
  parsed <- parse_text_predictions(
    c("Summary: three immune populations", "0: T cells", "1: B cells", "2: NK cells"),
    all_clusters = c("0", "1", "2")
  )
  expect_identical(parsed, list("0" = "T cells", "1" = "B cells", "2" = "NK cells"))
})

test_that("parse_text_predictions maps out-of-order labels by cluster ID, ignoring hallucinated ones", {
  parsed <- parse_text_predictions(
    c("2: NK cells", "1: B cells", "0: T cells", "5: Doublets"),
    all_clusters = c("0", "1", "2")
  )
  expect_identical(parsed, list("0" = "T cells", "1" = "B cells", "2" = "NK cells"))
})

test_that("parse_text_predictions does not strip a hyphenated cell type as an ordinal", {
  parsed <- parse_text_predictions(
    c("5 - HT neurons"),
    all_clusters = c("0")
  )
  expect_identical(parsed, list("0" = "5 - HT neurons"))
})

test_that("compare_model_predictions pads unequal lengths without recycling", {
  res <- testthat::with_mocked_bindings({
    capture.output({
      result <- suppressMessages(compare_model_predictions(
        input = list(
          "0" = list(genes = c("G1")),
          "1" = list(genes = c("G2")),
          "2" = list(genes = c("G3"))
        ),
        tissue_name = "PBMC",
        models = c("m1", "m2"),
        api_keys = list(openai = "k")
      ))
    })
    result
  },
  get_api_key = function(model, api_keys) "k",
  get_provider = function(model) "openai",
  annotate_cell_types = function(input, tissue_name, model, api_key, top_gene_count = 10, ...) {
    if (identical(model, "m1")) c("A", "B", "C") else c("A", "B")
  },
  standardize_cell_type_names = function(predictions, models, api_keys, standardization_model = "x", base_urls = NULL) {
    predictions
  })

  expect_identical(dim(res$comparison_matrix), c(3L, 2L))
  expect_true(is.na(res$comparison_matrix[3, "m2"]))
  expect_true(is.na(res$consensus_predictions[3]))
  expect_true(is.na(res$consensus_proportions[3]))
  expect_true(is.na(res$entropies[3]))
  expect_identical(rownames(res$comparison_matrix), c("0", "1", "2"))
  expect_identical(names(res$consensus_predictions), c("0", "1", "2"))
  expect_false(is.nan(res$summary_stats$mean_consensus_proportion))
  expect_false(is.nan(res$summary_stats$mean_entropy))
})

test_that("compare_model_predictions aligns explicit labels by cluster ID", {
  res <- testthat::with_mocked_bindings({
    capture.output({
      result <- suppressMessages(compare_model_predictions(
        input = list(
          alpha = list(genes = c("G1")),
          beta = list(genes = c("G2"))
        ),
        tissue_name = "PBMC",
        models = c("m1", "m2"),
        api_keys = list(openai = "k")
      ))
    })
    result
  },
  get_api_key = function(model, api_keys) "k",
  get_provider = function(model) "openai",
  annotate_cell_types = function(input, tissue_name, model, api_key, top_gene_count = 10, ...) {
    if (identical(model, "m1")) {
      c("beta: B cell", "alpha: T cell")
    } else {
      c("alpha: T cell", "beta: B cell")
    }
  },
  standardize_cell_type_names = function(predictions, models, api_keys, standardization_model = "x", base_urls = NULL) {
    predictions
  })

  expect_identical(rownames(res$comparison_matrix), c("alpha", "beta"))
  expect_identical(unname(res$comparison_matrix[, "m1"]), c("T cell", "B cell"))
  expect_identical(unname(res$consensus_predictions), c("T cell", "B cell"))
})

test_that("compare_model_predictions does not report a tied vote as consensus", {
  result <- testthat::with_mocked_bindings({
    capture.output({
      comparison <- suppressMessages(compare_model_predictions(
        input = list("0" = "G1"),
        tissue_name = "PBMC",
        models = c("m1", "m2"),
        api_keys = list(openai = "key"),
        consensus_threshold = 0.5
      ))
    })
    comparison
  },
  get_api_key = function(...) "key",
  get_provider = function(...) "openai",
  annotate_cell_types = function(input, tissue_name, model, ...) {
    if (model == "m1") "B cell" else "T cell"
  },
  standardize_cell_type_names = function(predictions, ...) predictions)

  expect_true(is.na(result$consensus_predictions[["0"]]))
  expect_true(is.na(result$consensus_proportions[["0"]]))
})

test_that("align_model_predictions honors names and marks missing clusters", {
  aligned <- align_model_predictions(
    c(beta = "B cell", alpha = "T cell"),
    c("alpha", "beta", "gamma")
  )

  expect_identical(
    aligned,
    c(alpha = "T cell", beta = "B cell", gamma = NA_character_)
  )
})

test_that("align_model_predictions treats internal sentinels as missing", {
  aligned <- align_model_predictions(
    c("0: Unknown", "1: Prediction_Missing", "2: T cell"),
    c("0", "1", "2")
  )

  expect_true(is.na(aligned[["0"]]))
  expect_true(is.na(aligned[["1"]]))
  expect_identical(aligned[["2"]], "T cell")
})

test_that("consensus phases share canonical numeric cluster order", {
  input <- list("10" = c("G10"), "2" = c("G2"), "1" = c("G1"))
  predictions <- list(
    m1 = c("Type 1", "Type 2", "Type 10"),
    m2 = c("Type 1", "Type 2", "Type 10")
  )

  result <- testthat::with_mocked_bindings({
    suppressMessages(identify_controversial_clusters(
      input = input,
      individual_predictions = predictions,
      controversy_threshold = 0.5,
      entropy_threshold = 1,
      api_keys = list(openai = "k")
    ))
  },
  check_consensus = function(round_responses, ...) {
    list(
      reached = TRUE,
      consensus_proportion = 1,
      entropy = 0,
      majority_prediction = unname(round_responses[[1]])
    )
  },
  log_info = function(...) NULL,
  log_warn = function(...) NULL)

  expect_identical(
    result$final_annotations,
    list("1" = "Type 1", "2" = "Type 2", "10" = "Type 10")
  )
})

test_that("standardize_cell_type_names ignores prose and accepts numbered mapping keys", {
  result <- testthat::with_mocked_bindings({
    suppressMessages(standardize_cell_type_names(
      predictions = list(m1 = c("1: T cell", "B cell")),
      models = "m1",
      api_keys = list(openai = "k"),
      standardization_model = "gpt-5.5"
    ))
  },
  get_api_key = function(...) "k",
  get_model_response = function(...) c(
    "Here is the requested mapping:",
    "1: T cell: CD4 T cell",
    "B cell: B cell",
    "Note: keep broad immune labels unchanged"
  ),
  log_warn = function(...) NULL)

  expect_identical(result$m1, c("CD4 T cell", "B cell"))
})

test_that("standardization defaults to the first requested model", {
  requested_model <- NULL
  result <- testthat::with_mocked_bindings(
    standardize_cell_type_names(
      predictions = list(m1 = "T cell", m2 = "T cell"),
      models = c("m1", "m2"),
      api_keys = list(openai = "key", anthropic = "other-key")
    ),
    get_api_key = function(...) "key",
    get_model_response = function(prompt, model, ...) {
      requested_model <<- model
      "T cell: T cell"
    },
    log_warn = function(...) NULL
  )

  expect_identical(requested_model, "m1")
  expect_identical(result, list(m1 = "T cell", m2 = "T cell"))
})

test_that("compare_model_predictions requires two eligible models", {
  expect_error(
    suppressWarnings(compare_model_predictions(
      input = list("0" = "G1"),
      tissue_name = "PBMC",
      models = c("gpt-5.5", "bad-model"),
      api_keys = list(openai = "key")
    )),
    "At least two models"
  )
})

test_that("compare_model_predictions skips invalid models and uses valid remaining models", {
  res <- testthat::with_mocked_bindings({
    capture.output({
      result <- suppressWarnings(suppressMessages(compare_model_predictions(
        input = list("0" = list(genes = c("G1"))),
        tissue_name = "PBMC",
        models = c("bad-model", "m1", "m2"),
        api_keys = list(openai = "k")
      )))
    })
    result
  },
  get_api_key = function(model, api_keys) {
    if (identical(model, "bad-model")) stop("unknown model") else "k"
  },
  get_provider = function(model) {
    if (identical(model, "bad-model")) stop("unknown model") else "openai"
  },
  annotate_cell_types = function(input, tissue_name, model, api_key, top_gene_count = 10, ...) {
    c("T cell")
  },
  standardize_cell_type_names = function(predictions, models, api_keys, standardization_model = "x", base_urls = NULL) {
    predictions
  })

  expect_identical(colnames(res$comparison_matrix), c("m1", "m2"))
})

test_that("interactive_consensus_annotation supports mapped cluster selectors", {
  mocked_call <- function(selector) {
    testthat::with_mocked_bindings({
      suppressMessages(interactive_consensus_annotation(
        input = list(
          "1" = c("CD3D"),
          "2" = list(genes = c("MS4A1"))
        ),
        tissue_name = "PBMC",
        models = c("gpt-5.5", "grok-4.3"),
        api_keys = list(openai = "k", grok = "k"),
        use_cache = FALSE,
        clusters_to_analyze = selector
      ))
    },
    log_info = function(...) NULL,
    print_consensus_summary = function(...) NULL,
    get_initial_predictions = function(input, ...) {
      # Predictions keyed by original cluster names (no 0-based conversion)
      cluster_ids <- names(input)
      preds <- setNames(rep("T", length(cluster_ids)), cluster_ids)
      list(
        individual_predictions = list(m1 = as.list(preds), m2 = as.list(preds)),
        successful_models = c("m1", "m2")
      )
    },
    identify_controversial_clusters = function(input, ...) {
      list(
        consensus_results = list(),
        controversial_clusters = character(0),
        final_annotations = setNames(rep("X", length(input)), names(input))
      )
    },
    process_controversial_clusters = function(...) {
      list(discussion_logs = list(), final_annotations = list())
    },
    combine_results = function(initial_results, controversy_results, discussion_results) {
      list(final_annotations = controversy_results$final_annotations)
    })
  }

  # Cluster names are preserved as-is (no 0-based conversion)
  result_single <- mocked_call("1")
  expect_identical(names(result_single$final_annotations), "1")

  result_both <- mocked_call(c("1", "2"))
  expect_identical(sort(names(result_both$final_annotations)), c("1", "2"))
})

test_that("parse_consensus_response falls back to flexible parsing for short responses", {
  result <- testthat::with_mocked_bindings({
    parse_consensus_response(
      c(
        "Consensus Proportion = 0.75",
        "Entropy = 0.81",
        "B cell"
      )
    )
  },
  get_logger = function() {
    list(
      warn = function(...) NULL,
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_identical(result$reached, FALSE)
  expect_equal(result$consensus_proportion, 0.75)
  expect_equal(result$entropy, 0.81)
  expect_identical(result$majority_prediction, "B cell")
})

test_that("get_initial_predictions skips invalid model names without aborting", {
  result <- testthat::with_mocked_bindings({
    suppressWarnings(get_initial_predictions(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      models = c("bad-model", "gpt-5.5"),
      api_keys = list(openai = "k"),
      top_gene_count = 10
    ))
  },
  get_provider = function(model) {
    if (identical(model, "bad-model")) stop("unknown model")
    "openai"
  },
  get_api_key = function(model, api_keys) {
    provider <- get_provider(model)
    if (provider %in% names(api_keys)) api_keys[[provider]] else NULL
  },
  annotate_cell_types = function(...) c("T cell"),
  log_info = function(...) NULL,
  log_warn = function(...) NULL)

  expect_identical(result$successful_models, "gpt-5.5")
  expect_true("gpt-5.5" %in% names(result$individual_predictions))
})

test_that("interactive_consensus_annotation consumes log_dir via initialize_logger", {
  seen_log_dir <- NULL

  testthat::with_mocked_bindings({
    suppressMessages(interactive_consensus_annotation(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      models = c("gpt-5.5", "grok-4.3"),
      api_keys = list(openai = "k", grok = "k"),
      use_cache = FALSE,
      log_dir = "custom_logs_dir"
    ))
  },
  initialize_logger = function(log_dir = "logs") {
    seen_log_dir <<- log_dir
    invisible(NULL)
  },
  log_info = function(...) NULL,
  print_consensus_summary = function(...) NULL,
  get_initial_predictions = function(input, ...) {
    list(
      individual_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      successful_models = c("m1", "m2")
    )
  },
  identify_controversial_clusters = function(input, ...) {
    list(
      consensus_results = list(),
      controversial_clusters = character(0),
      final_annotations = setNames(rep("X", length(input)), names(input))
    )
  },
  process_controversial_clusters = function(...) {
    list(discussion_logs = list(), final_annotations = list())
  },
  combine_results = function(initial_results, controversy_results, discussion_results) {
    list(final_annotations = controversy_results$final_annotations)
  })

  expect_identical(seen_log_dir, "custom_logs_dir")
})

test_that("combine_results provides stable aliases for return fields", {
  result <- testthat::with_mocked_bindings({
    combine_results(
      initial_results = list(individual_predictions = list(m1 = "A")),
      controversy_results = list(
        consensus_results = list("0" = list(reached = TRUE)),
        controversial_clusters = c("1"),
        final_annotations = list("0" = "T cell")
      ),
      discussion_results = list(
        discussion_logs = list("1" = list(rounds = list())),
        final_annotations = list("1" = "B cell")
      )
    )
  },
  get_logger = function() list(session_id = "session-test"))

  expect_identical(result$voting_results, result$initial_results)
  expect_identical(result$discussion_results, result$discussion_logs)
  expect_identical(result$final_consensus, result$final_annotations)
})

test_that("annotation cleanup is scalar-safe and consistent across result phases", {
  expect_identical(clean_annotation(c("T cell", "B cell")), "Unknown")
  expect_identical(clean_annotation(1), "Unknown")
  expect_identical(clean_annotation("cell type: T cell"), "T cell")
  expect_identical(clean_annotation("unknown"), "Unknown")

  result <- testthat::with_mocked_bindings(
    combine_results(
      initial_results = list(individual_predictions = list()),
      controversy_results = list(
        consensus_results = list(),
        controversial_clusters = "1",
        final_annotations = list("0" = "CELL TYPE: T cell")
      ),
      discussion_results = list(
        discussion_logs = list(),
        final_annotations = list("1" = c("B cell", "NK cell"))
      )
    ),
    get_logger = function() list(session_id = "session-test")
  )

  expect_identical(result$final_annotations[["0"]], "T cell")
  expect_identical(result$final_annotations[["1"]], "Unknown")
})

test_that("select_best_prediction uses model order rather than label length for ties", {
  result <- select_best_prediction(
    list(majority_prediction = "Parsing_Failed"),
    c("T cell", "longer unsupported label")
  )

  expect_identical(result, "T cell")
})

capture_discussion_genes <- function(input, cluster_id = "0", top_gene_count = 5) {
  captured_cluster_genes <- NULL
  testthat::with_mocked_bindings({
    suppressWarnings(suppressMessages(facilitate_cluster_discussion(
      cluster_id = cluster_id,
      input = input,
      tissue_name = "PBMC",
      models = c("m1", "m2"),
      api_keys = list(openai = "k"),
      initial_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      top_gene_count = top_gene_count,
      max_rounds = 2
    )))
  },
  create_initial_discussion_prompt = function(cluster_id, cluster_genes, tissue_name, initial_predictions) {
    captured_cluster_genes <<- cluster_genes
    "prompt"
  },
  get_model_response = function(...) "CELL TYPE: T cell",
  check_consensus = function(...) {
    list(reached = TRUE, consensus_proportion = 1, entropy = 0, majority_prediction = "T cell")
  },
  get_api_key = function(model, api_keys) "k",
  get_provider = function(model) "openai",
  get_logger = function() {
    list(log_discussion = function(...) NULL)
  },
  log_warn = function(...) NULL,
  log_info = function(...) NULL)
  captured_cluster_genes
}

test_that("facilitate_cluster_discussion extracts data-frame genes by marker rank", {
  markers <- data.frame(
    cluster = c(0, 0, 0, 1),
    gene = c("low", "top", "negative", "other"),
    avg_log2FC = c(1, 3, -2, 5)
  )

  expect_identical(capture_discussion_genes(markers, cluster_id = "0", top_gene_count = 2), "top,low")
})

test_that("facilitate_cluster_discussion preserves data-frame row order without avg_log2FC", {
  markers <- data.frame(
    cluster = c("0", "0", "0"),
    gene = c("G1", "G2", "G3")
  )

  expect_identical(capture_discussion_genes(markers, cluster_id = "0", top_gene_count = 2), "G1,G2")
})

test_that("facilitate_cluster_discussion keeps fallback cluster_genes on extraction error", {
  captured_cluster_genes <- capture_discussion_genes(42, cluster_id = "0")
  expect_true(grepl("Error extracting genes", captured_cluster_genes, fixed = TRUE))
})

test_that("facilitate_cluster_discussion falls back for missing clusters and malformed target entries", {
  missing_cluster_genes <- capture_discussion_genes(list("1" = c("G1")), cluster_id = "0")
  malformed_entry_genes <- capture_discussion_genes(list("0" = list(markers = c("G1"))), cluster_id = "0")

  expect_true(grepl("Cluster '0' was not found", missing_cluster_genes, fixed = TRUE))
  expect_true(grepl("character vector of genes", malformed_entry_genes, fixed = TRUE))
})


test_that("facilitate_cluster_discussion ignores malformed unrelated list entries", {
  cluster_genes <- capture_discussion_genes(
    list("0" = c("G1", "G2"), "1" = list(markers = c("bad"))),
    cluster_id = "0"
  )

  expect_identical(cluster_genes, "G1,G2")
})

test_that("is_error_response flags pure errors but preserves partial annotations", {
  expect_true(is_error_response("Error: request failed"))
  expect_true(is_error_response("  ERROR: request failed"))
  # Every non-empty line is an error -> the whole response is an error.
  expect_true(is_error_response(c("Error: rate limit", "error : try again")))
  # A usable annotation alongside an error marker is NOT a pure error: a single
  # flagged/uncertain cluster must not discard the model's other annotations.
  expect_false(is_error_response(c("CD8+ T cells", "Error: markers ambiguous", "B cells")))
  expect_false(is_error_response("CELL TYPE: T cell"))
  expect_false(is_error_response(list(message = "Error: request failed")))
  expect_false(is_error_response(NA_character_))
  expect_false(is_error_response(character(0)))
})

test_that("model response normalization gives every provider the same line contract", {
  expect_identical(
    normalize_model_response_lines(c("  Cluster 1: T cells,  ", "", "Cluster 2: B cells\n")),
    c("Cluster 1: T cells", "Cluster 2: B cells")
  )

  invalid_responses <- list(NULL, character(), NA_character_, 123, "  ", "Error: failed")
  for (response in invalid_responses) {
    expect_error(normalize_model_response_lines(response), "Model response")
    expect_false(is_valid_model_response(response))
  }
})

test_that("token usage normalization rejects malformed external telemetry", {
  expect_identical(
    normalize_token_usage(list(
      prompt_tokens = 0,
      completion_tokens = 2L,
      total_tokens = 2,
      cost = 1
    )),
    list(
      prompt_tokens = 0,
      completion_tokens = 2,
      total_tokens = 2,
      cost = 1
    )
  )

  invalid <- normalize_token_usage(list(
    prompt_tokens = -1,
    completion_tokens = TRUE,
    total_tokens = "3",
    cost = Inf
  ))
  expect_null(invalid$prompt_tokens)
  expect_null(invalid$completion_tokens)
  expect_null(invalid$total_tokens)
  expect_false("cost" %in% names(invalid))
  expect_null(normalize_token_usage("not usage"))
})

test_that("provider usage extractors share one canonical schema", {
  openai_response <- make_test_json_response(list(
    usage = list(
      prompt_tokens = 10,
      completion_tokens = 4,
      total_tokens = 14,
      cost = 0.001
    )
  ))
  anthropic_response <- make_test_json_response(list(
    usage = list(input_tokens = 10, output_tokens = 4)
  ))
  gemini_response <- make_test_json_response(list(
    usageMetadata = list(
      promptTokenCount = 10,
      candidatesTokenCount = 4,
      totalTokenCount = 14
    )
  ))

  expect_identical(
    OpenAIProcessor$new()$extract_usage(openai_response),
    list(prompt_tokens = 10, completion_tokens = 4, total_tokens = 14, cost = 0.001)
  )
  expect_identical(
    AnthropicProcessor$new()$extract_usage(anthropic_response),
    list(prompt_tokens = 10, completion_tokens = 4, total_tokens = 14)
  )
  expect_identical(
    GeminiProcessor$new()$extract_usage(gemini_response),
    list(prompt_tokens = 10, completion_tokens = 4, total_tokens = 14)
  )
})

test_that("BaseAPIProcessor forwards normalized usage to API call logs", {
  captured_tokens <- NULL
  logger <- list(
    info = function(...) NULL,
    debug = function(...) NULL,
    warn = function(...) NULL,
    error = function(...) NULL,
    log_api_request_response = function(...) NULL,
    log_api_call = function(provider, model, duration, success = TRUE, tokens = NULL) {
      captured_tokens <<- tokens
    }
  )

  testthat::with_mocked_bindings({
    UsageProcessor <- R6::R6Class(
      "UsageProcessor",
      inherit = BaseAPIProcessor,
      public = list(
        initialize = function() super$initialize("openai", NULL),
        get_default_api_url = function() "https://example.test",
        make_api_call = function(chunk_content, model, api_key) {
          make_test_json_response(list(
            usage = list(
              prompt_tokens = 5,
              completion_tokens = 2,
              total_tokens = 7
            )
          ))
        },
        extract_response_content = function(response, model) "T cell"
      )
    )

    expect_identical(
      UsageProcessor$new()$process_request("prompt", "model", "key"),
      "T cell"
    )
  }, get_logger = function() logger)

  expect_identical(
    captured_tokens,
    list(prompt_tokens = 5, completion_tokens = 2, total_tokens = 7)
  )
})

test_that("filter_valid_responses excludes case-insensitive error sentinels", {
  result <- testthat::with_mocked_bindings({
    filter_valid_responses(
      list(
        good = "CELL TYPE: T cell",
        bad_upper = "ERROR: provider failed",
        bad_spaced = " error : provider failed",
        missing = NA_character_,
        blank = "   ",
        empty = character(0)
      ),
      cluster_id = "0"
    )
  },
  log_warn = function(...) NULL)

  expect_identical(names(result), "good")
})

test_that("facilitate_cluster_discussion selects the current cluster prediction", {
  captured_predictions <- NULL

  testthat::with_mocked_bindings({
    suppressMessages(facilitate_cluster_discussion(
      cluster_id = "1",
      input = list("0" = c("G1"), "1" = c("G2")),
      tissue_name = "PBMC",
      models = c("m1", "m2"),
      api_keys = list(openai = "k"),
      initial_predictions = list(
        m1 = c("T cell", "B cell"),
        m2 = c("T cell", "B cell")
      ),
      top_gene_count = 5,
      max_rounds = 1
    ))
  },
  create_initial_discussion_prompt = function(cluster_id, cluster_genes, tissue_name,
                                               initial_predictions) {
    captured_predictions <<- initial_predictions
    "prompt"
  },
  get_model_response = function(...) "CELL TYPE: B cell",
  check_consensus = function(...) {
    list(reached = TRUE, consensus_proportion = 1, entropy = 0, majority_prediction = "B cell")
  },
  get_api_key = function(...) "k",
  get_logger = function() list(log_discussion = function(...) NULL),
  log_info = function(...) NULL,
  log_warn = function(...) NULL)

  expect_identical(captured_predictions, list(m1 = "B cell", m2 = "B cell"))
})

test_that("initial discussion prompt renders canonical scalar predictions", {
  prompt <- create_initial_discussion_prompt(
    cluster_id = "1",
    cluster_genes = "MS4A1,CD79A",
    tissue_name = "PBMC",
    initial_predictions = list(m1 = "B cell", m2 = "Memory B cell")
  )

  expect_match(prompt, "m1: B cell", fixed = TRUE)
  expect_match(prompt, "m2: Memory B cell", fixed = TRUE)
  expect_false(grepl("No prediction", prompt, fixed = TRUE))
})

test_that("discussion prompt preserves multi-line responses per model", {
  prompt <- create_discussion_prompt(
    cluster_id = "1",
    cluster_genes = "MS4A1,CD79A",
    tissue_name = "PBMC",
    previous_rounds = list(list(
      round_number = 1,
      responses = list(
        m1 = c("CELL TYPE: B cell", "GROUNDS: MS4A1"),
        m2 = c("CELL TYPE: B cell", "GROUNDS: CD79A")
      )
    )),
    round_number = 2
  )

  expect_match(prompt, "m1:\nCELL TYPE: B cell\nGROUNDS: MS4A1", fixed = TRUE)
  expect_match(prompt, "m2:\nCELL TYPE: B cell\nGROUNDS: CD79A", fixed = TRUE)
  expect_false(grepl('c\\("CELL TYPE', prompt))
})

test_that("annotate_cell_types supports custom providers", {
  custom_env <- new.env(parent = emptyenv())
  assign("customx", TRUE, envir = custom_env)

  result <- testthat::with_mocked_bindings({
    annotate_cell_types(
      input = list("0" = list(genes = c("CD3D", "CD3E"))),
      tissue_name = "PBMC",
      model = "custom-model",
      api_key = "k"
    )
  },
  get_provider = function(model) "customx",
  resolve_provider_base_url = function(provider, base_urls) NULL,
  create_annotation_prompt = function(input, tissue_name, top_gene_count = 10) {
    list(prompt = "prompt", expected_count = 1L, gene_lists = list("0" = "CD3D, CD3E"))
  },
  process_custom = function(prompt, model, api_key) "custom-ok",
  custom_providers = custom_env,
  log_info = function(...) NULL,
  log_debug = function(...) NULL)

  expect_identical(result, "custom-ok")
})

test_that("get_model_response enforces the response contract for custom providers", {
  custom_env <- new.env(parent = emptyenv())
  assign("customx", TRUE, envir = custom_env)
  invalid_responses <- list(NULL, "", NA_character_, 123, "Error: failed")

  for (invalid_response in invalid_responses) {
    expect_error(
      testthat::with_mocked_bindings(
        get_model_response("prompt", "custom-model", "key"),
        get_provider = function(...) "customx",
        resolve_provider_base_url = function(...) NULL,
        process_custom = function(...) invalid_response,
        custom_providers = custom_env,
        log_debug = function(...) NULL
      ),
      "Invalid response format from provider 'customx'"
    )
  }
})

test_that("get_model_response normalizes built-in and custom provider output identically", {
  custom_env <- new.env(parent = emptyenv())
  assign("customx", TRUE, envir = custom_env)

  result <- testthat::with_mocked_bindings(
    get_model_response("prompt", "custom-model", "key"),
    get_provider = function(...) "customx",
    resolve_provider_base_url = function(...) NULL,
    process_custom = function(...) c("  Cluster 1: T cells,", "Cluster 2: B cells\n"),
    custom_providers = custom_env,
    log_debug = function(...) NULL
  )

  expect_identical(result, c("Cluster 1: T cells", "Cluster 2: B cells"))
})

test_that("process_custom forwards model_config when provider supports it", {
  skip_if_not_installed("testthat", minimum_version = "3.2.0")
  original_providers <- as.list(custom_providers)
  original_models <- as.list(custom_models)
  rm(list = ls(envir = custom_providers), envir = custom_providers)
  rm(list = ls(envir = custom_models), envir = custom_models)
  on.exit({
    rm(list = ls(envir = custom_providers), envir = custom_providers)
    rm(list = ls(envir = custom_models), envir = custom_models)
    list2env(original_providers, envir = custom_providers)
    list2env(original_models, envir = custom_models)
  }, add = TRUE)

  captured_config <- NULL
  assign("customx",
         list(
           process_fn = function(prompt, model, api_key, model_config) {
             captured_config <<- model_config
             "custom-ok"
           },
           description = NULL,
           models = "custom-model"
         ),
         envir = custom_providers)
  assign("custom-model",
         list(provider = "customx", config = list(temperature = 0.2, max_tokens = 100)),
         envir = custom_models)

  result <- testthat::with_mocked_bindings({
    suppressMessages(process_custom("prompt", "custom-model", "k"))
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_identical(result, "custom-ok")
  expect_identical(captured_config, list(temperature = 0.2, max_tokens = 100))

  captured_config <- NULL
  custom_providers$customx$process_fn <- function(prompt, model, api_key, ...) {
    captured_config <<- list(...)[["model_config"]]
    "custom-ok"
  }

  result <- testthat::with_mocked_bindings({
    suppressMessages(process_custom("prompt", "custom-model", "k"))
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_identical(result, "custom-ok")
  expect_identical(captured_config, list(temperature = 0.2, max_tokens = 100))

  captured_base_url <- NULL
  custom_providers$customx$process_fn <- function(prompt, model, api_key,
                                                   model_config, base_url) {
    captured_config <<- model_config
    captured_base_url <<- base_url
    "custom-ok"
  }

  result <- testthat::with_mocked_bindings({
    suppressMessages(process_custom(
      "prompt",
      "custom-model",
      "k",
      "https://custom.example.test/v1"
    ))
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_identical(result, "custom-ok")
  expect_identical(captured_config, list(temperature = 0.2, max_tokens = 100))
  expect_identical(captured_base_url, "https://custom.example.test/v1")

  custom_providers$customx$process_fn <- function(prompt, model, api_key) {
    stop_api_request_error("rate limited", status_code = 429L)
  }
  api_error <- tryCatch(
    suppressMessages(process_custom("prompt", "custom-model", "k")),
    error = identity
  )
  expect_s3_class(api_error, "mllm_api_error")
  expect_identical(api_error$status_code, 429L)
  expect_true(api_error$retryable)

  custom_providers$customx$process_fn <- function(prompt, model, api_key) "custom-ok"
  expect_error(
    suppressMessages(process_custom(
      "prompt",
      "custom-model",
      "k",
      "https://custom.example.test/v1"
    )),
    "does not accept base_url"
  )
})

test_that("custom provider identifiers are normalized and cannot shadow built-ins", {
  expect_error(
    register_custom_provider(" openai ", function(prompt, model, api_key) "x"),
    "reserved for a built-in provider"
  )
  expect_error(
    register_custom_provider("openrouter", function(prompt, model, api_key) "x"),
    "reserved for a built-in provider"
  )
  expect_error(
    register_custom_provider("   ", function(prompt, model, api_key) "x"),
    "provider_name must be a non-empty character scalar"
  )
  expect_error(
    register_custom_model(NA_character_, "customx"),
    "model_name must be a non-empty character scalar"
  )
  expect_error(
    register_custom_provider("../unsafe", function(prompt, model, api_key) "x"),
    "must contain only letters, digits"
  )
})

test_that("check_consensus normalizes discussion cell type labels", {
  result <- testthat::with_mocked_bindings({
    suppressMessages(check_consensus(
      list(
        m1 = "CELL TYPE: T cell",
        m2 = "Reasoning\nCELL TYPE: T cell"
      ),
      controversy_threshold = 1,
      entropy_threshold = 0
    ))
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      debug = function(...) NULL,
      warn = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_true(result$reached)
  expect_identical(result$majority_prediction, "T cell")
})

test_that("check_consensus rejects unparseable or missing responses", {
  result <- testthat::with_mocked_bindings({
    check_consensus(
      list(
        m1 = "Reasoning without a final label",
        m2 = "More reasoning\nwithout a final label",
        m3 = NA_character_
      )
    )
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      debug = function(...) NULL,
      warn = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_false(result$reached)
  expect_identical(result$majority_prediction, "Insufficient_Responses")
})


test_that("consensus prompt documents inclusive consensus threshold", {
  prompt <- create_consensus_check_prompt(
    c("T cell", "B cell"),
    controversy_threshold = 0.5,
    entropy_threshold = 1
  )

  expect_true(grepl("Consensus Proportion >= 0.5", prompt, fixed = TRUE))
  expect_true(grepl("Preserve biologically meaningful subtype or state qualifiers", prompt, fixed = TRUE))
  expect_true(grepl("do not merge labels when doing so would lose specificity", prompt, fixed = TRUE))
  expect_false(grepl("ignoring differences in.*Additional qualifiers", prompt))
})


test_that("parse_consensus_response handles NA_character_ safely", {
  result <- testthat::with_mocked_bindings({
    parse_consensus_response(NA_character_)
  },
  get_logger = function() {
    list(
      warn = function(...) NULL,
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })

  expect_identical(result, list(
    reached = FALSE,
    consensus_proportion = 0,
    entropy = 0,
    majority_prediction = "Unknown"
  ))
})

test_that("print_consensus_summary handles vector predictions without crashing", {
  results <- list(
    final_annotations = list("0" = c("Consensus A", "Consensus B")),
    controversial_clusters = c("0"),
    discussion_logs = list(
      "0" = list(initial_predictions = list(m1 = c("Pred A", "Pred B")))
    )
  )

  expect_no_error(capture.output(print_consensus_summary(results)))
})

test_that("QwenProcessor caches endpoint selection per API key", {
  original_cache <- as.list(.qwen_endpoint_cache)
  rm(list = ls(envir = .qwen_endpoint_cache), envir = .qwen_endpoint_cache)
  on.exit({
    rm(list = ls(envir = .qwen_endpoint_cache), envir = .qwen_endpoint_cache)
    list2env(original_cache, envir = .qwen_endpoint_cache)
  }, add = TRUE)

  key_a <- digest::digest("key-a", algo = "xxhash64")
  key_b <- digest::digest("key-b", algo = "xxhash64")
  assign(key_a, "https://intl.example.test", envir = .qwen_endpoint_cache)
  assign(key_b, "https://domestic.example.test", envir = .qwen_endpoint_cache)

  processor <- QwenProcessor$new()

  expect_identical(processor$get_working_api_url("key-a"), "https://intl.example.test")
  expect_identical(processor$get_working_api_url("key-b"), "https://domestic.example.test")
})

test_that("QwenProcessor does not cache an unverified fallback endpoint", {
  original_cache <- as.list(.qwen_endpoint_cache)
  rm(list = ls(envir = .qwen_endpoint_cache), envir = .qwen_endpoint_cache)
  on.exit({
    rm(list = ls(envir = .qwen_endpoint_cache), envir = .qwen_endpoint_cache)
    list2env(original_cache, envir = .qwen_endpoint_cache)
  }, add = TRUE)

  testthat::with_mocked_bindings({
    ProbeFailingQwenProcessor <- R6::R6Class(
      "ProbeFailingQwenProcessor",
      inherit = QwenProcessor,
      public = list(probe_count = 0L),
      private = list(
        test_endpoint = function(url, api_key) {
          self$probe_count <- self$probe_count + 1L
          FALSE
        }
      )
    )

    processor <- ProbeFailingQwenProcessor$new()
    expected <- processor$get_default_api_url()
    expect_identical(processor$get_working_api_url("key-a"), expected)
    expect_identical(processor$get_working_api_url("key-a"), expected)
    expect_identical(processor$probe_count, 6L)
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      warn = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })
})

test_that("resolve_provider_base_url matches Python URL contract", {
  expect_null(resolve_provider_base_url("openai", NULL))
  expect_null(resolve_provider_base_url("openai", list(anthropic = "https://anthropic.example.test/v1")))

  expect_identical(
    resolve_provider_base_url("  OPENAI  ", "  https://proxy.example.test/v1/  "),
    "https://proxy.example.test/v1"
  )
  expect_identical(
    resolve_provider_base_url(
      "openai",
      list("  OpenAI  " = "https://openai-proxy.example.test/v1/chat/completions/")
    ),
    "https://openai-proxy.example.test/v1/chat/completions"
  )

  expect_error(resolve_provider_base_url("openai", "not-a-url"), "Invalid base URL")
  expect_error(resolve_provider_base_url("openai", "https://api.example.test/v1?token=abc"), "Invalid base URL")
  expect_error(resolve_provider_base_url("openai", list(openai = 123)), "must be a string URL")
  expect_error(resolve_provider_base_url("openai", c("https://a.example", "https://b.example")), "base_urls must be")
  expect_error(
    resolve_provider_base_url("openai", list("OpenAI" = "https://a.example", " openai " = "https://b.example")),
    "Ambiguous base_urls provider names"
  )
  expect_error(resolve_provider_base_url("openai", list("https://a.example")), "named list")
  expect_error(resolve_provider_base_url("openai", setNames(list("https://a.example"), "")), "non-empty strings")

  invalid_ports <- c(
    "https://api.example.test:invalid/v1",
    "https://api.example.test:0/v1",
    "https://api.example.test:99999/v1",
    "https://api.example.test:/v1"
  )
  for (url in invalid_ports) {
    expect_false(validate_base_url(url))
  }
  expect_true(validate_base_url("https://api.example.test:65535/v1"))
})

test_that("provider processors normalize and validate direct base URLs", {
  processor <- OpenAIProcessor$new("  https://proxy.example.test/v1/chat/completions/  ")
  expect_identical(
    processor$get_api_url(),
    "https://proxy.example.test/v1/chat/completions"
  )

  invalid_urls <- list(123, character(), NA_character_, "not-a-url")
  for (base_url in invalid_urls) {
    expect_error(OpenAIProcessor$new(base_url), "base_urls|Invalid base URL")
  }
})

test_that("retryable HTTP status policy rejects malformed status values", {
  expect_true(all(vapply(c(408, 425, 429, 500, 503), is_retryable_http_status, logical(1))))
  invalid_statuses <- list(NULL, TRUE, NA_real_, NaN, Inf, 500.5, 200, 302, 400, 404)
  expect_false(any(vapply(invalid_statuses, is_retryable_http_status, logical(1))))
})

test_that("provider default endpoints use current OpenAI-compatible chat APIs", {
  expect_identical(
    QwenProcessor$new()$get_default_api_url(),
    "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
  )
  expect_identical(
    MinimaxProcessor$new()$get_default_api_url(),
    "https://api.minimax.io/v1/chat/completions"
  )
})


test_that("BaseAPIProcessor enforces and normalizes required string inputs", {
  TP <- R6::R6Class(
    "TP",
    inherit = BaseAPIProcessor,
    public = list(
      get_default_api_url = function() "x",
      make_api_call = function(chunk_content, model, api_key) {
        expect_identical(chunk_content, "prompt")
        expect_identical(model, "model")
        expect_identical(api_key, "key")
        list()
      },
      extract_response_content = function(response, model) "ok"
    )
  )

  p <- TP$new("tp")
  expect_error(
    p$process_request(c("a", "b"), "m", "k"),
    "Prompt is required but not provided"
  )
  expect_error(p$process_request(123, "m", "k"), "Prompt is required but not provided")
  expect_error(p$process_request("p", 123, "k"), "Model is required but not provided")
  expect_error(p$process_request("p", "m", 123), "API key is required but not provided")
  expect_identical(p$process_request("  prompt  ", "  model  ", "  key  "), "ok")
})

test_that("BaseAPIProcessor accepts only HTTP 200 responses", {
  testthat::with_mocked_bindings({
    StatusCheckingProcessor <- R6::R6Class(
      "StatusCheckingProcessor",
      inherit = BaseAPIProcessor,
      public = list(
        initialize = function() super$initialize("openai", NULL),
        get_default_api_url = function() "https://example.test",
        make_api_call = function(chunk_content, model, api_key) NULL,
        extract_response_content = function(response, model) NULL,
        check_status = function(status_code) {
          private$stop_for_http_error(
            make_test_json_response(list(message = "redirected"), status_code),
            "gpt-5.5"
          )
        }
      )
    )

    processor <- StatusCheckingProcessor$new()
    expect_no_error(processor$check_status(200L))
    redirect_error <- tryCatch(
      processor$check_status(302L),
      error = identity
    )
    expect_s3_class(redirect_error, "mllm_api_error")
    expect_identical(redirect_error$status_code, 302L)
    expect_false(redirect_error$retryable)
    expect_identical(
      conditionMessage(redirect_error),
      "OpenAI API request failed: redirected"
    )
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })
})

test_that("BaseAPIProcessor rejects empty, missing, and error text responses", {
  invalid_results <- list("", "   ", NA_character_, character(0), "Error: failed")

  for (invalid_result in invalid_results) {
    testthat::with_mocked_bindings({
      InvalidResponseProcessor <- R6::R6Class(
        "InvalidResponseProcessor",
        inherit = BaseAPIProcessor,
        public = list(
          initialize = function() super$initialize("test", NULL),
          get_default_api_url = function() "https://example.test",
          make_api_call = function(chunk_content, model, api_key) list(),
          extract_response_content = function(response, model) invalid_result
        )
      )

      expect_error(
        InvalidResponseProcessor$new()$process_request("prompt", "model", "key"),
        "Invalid response format from API"
      )
    },
    get_logger = function() {
      list(
        info = function(...) NULL,
        debug = function(...) NULL,
        error = function(...) NULL,
        log_api_call = function(...) NULL,
        log_api_request_response = function(...) NULL
      )
    })
  }
})

test_that("BaseAPIProcessor does not duplicate provider HTTP error context", {
  testthat::with_mocked_bindings({
    FailingHTTPProcessor <- R6::R6Class(
      "FailingHTTPProcessor",
      inherit = BaseAPIProcessor,
      public = list(
        initialize = function() super$initialize("openai", NULL),
        get_default_api_url = function() "https://example.test",
        make_api_call = function(chunk_content, model, api_key) {
          private$stop_for_http_error(
            make_test_json_response(list(error = "bad request"), 400L),
            model,
            "OpenAI"
          )
        },
        extract_response_content = function(response, model) "unused"
      )
    )

    error <- tryCatch(
      FailingHTTPProcessor$new()$process_request("prompt", "model", "key"),
      error = identity
    )
    expect_s3_class(error, "mllm_api_error")
    expect_identical(error$status_code, 400L)
    expect_false(error$retryable)
    expect_identical(
      conditionMessage(error),
      "OpenAI API request failed: bad request"
    )
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL,
      log_api_call = function(...) NULL,
      log_api_request_response = function(...) NULL
    )
  })
})

test_that("BaseAPIProcessor shared chat completion helpers keep canonical contract", {
  testthat::with_mocked_bindings({
    HelperProcessor <- R6::R6Class(
      "HelperProcessor",
      inherit = BaseAPIProcessor,
      public = list(
        initialize = function() super$initialize("openai", NULL),
        get_default_api_url = function() "https://example.test/v1/chat/completions",
        make_api_call = function(chunk_content, model, api_key) NULL,
        extract_response_content = function(response, model) NULL,
        build_body_for_test = function() {
          private$build_chat_completions_body(
            "genes",
            "gpt-5.5",
            extra = list(stream = FALSE)
          )
        },
        extract_content_for_test = function(payload) {
          private$extract_chat_completions_content(
            make_test_json_response(payload),
            "gpt-5.5"
          )
        }
      )
    )

    processor <- HelperProcessor$new()
    body <- processor$build_body_for_test()
    expect_identical(body$model, "gpt-5.5")
    expect_identical(body$messages[[1]], list(role = "user", content = "genes"))
    expect_identical(body$stream, FALSE)
    expect_identical(
      processor$extract_content_for_test(list(
        choices = list(list(message = list(content = "T cell\nB cell")))
      )),
      "T cell\nB cell"
    )
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })
})

test_that("BaseAPIProcessor extracts provider error messages from common shapes", {
  testthat::with_mocked_bindings({
    HelperProcessor <- R6::R6Class(
      "HelperProcessor",
      inherit = BaseAPIProcessor,
      public = list(
        initialize = function() super$initialize("openai", NULL),
        get_default_api_url = function() "https://example.test/v1/chat/completions",
        make_api_call = function(chunk_content, model, api_key) NULL,
        extract_response_content = function(response, model) NULL,
        error_for_test = function(payload) {
          private$extract_error_message(make_test_json_response(payload, status_code = 400L))
        }
      )
    )

    processor <- HelperProcessor$new()
    expect_identical(
      processor$error_for_test(list(error = list(message = "nested error"))),
      "nested error"
    )
    expect_identical(
      processor$error_for_test(list(error = "flat error")),
      "flat error"
    )
    expect_identical(
      processor$error_for_test(list(message = "top-level error")),
      "top-level error"
    )
    expect_identical(
      processor$error_for_test(list(error = list(message = NA_character_))),
      "HTTP 400 error"
    )
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })
})

test_that("MiniMax business error without status message is stable", {
  response <- make_test_json_response(
    list(base_resp = list(status_code = 1001, status_msg = NA_character_))
  )

  testthat::with_mocked_bindings({
    expect_error(
      MinimaxProcessor$new()$extract_response_content(response, "MiniMax-M2.7"),
      "MiniMax API error: Unknown MiniMax API error"
    )
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })
})

test_that("MiniMax strips think blocks from OpenAI-compatible content", {
  response <- make_test_json_response(
    list(
      choices = list(list(message = list(
        content = "<think>internal reasoning</think>\nCELL TYPE: T cell"
      )))
    )
  )

  testthat::with_mocked_bindings({
    result <- MinimaxProcessor$new()$extract_response_content(response, "MiniMax-M2.7")
    expect_identical(result, "CELL TYPE: T cell")
  },
  get_logger = function() {
    list(
      info = function(...) NULL,
      debug = function(...) NULL,
      error = function(...) NULL
    )
  })
})

test_that("get_provider validates model as non-empty scalar", {
  expect_identical(get_provider("  gpt-5.5  "), "openai")
  expect_identical(get_provider("  anthropic/claude-opus-4.7  "), "openrouter")
  expect_error(get_provider(character(0)), "model must be a non-empty character scalar")
  expect_error(get_provider(c("gpt-5.5", "grok-4.3")), "model must be a non-empty character scalar")
  expect_error(get_provider(NA_character_), "model must be a non-empty character scalar")
  expect_error(get_provider("o2-preview"), "Cannot determine provider")
  expect_error(get_provider("o9-preview"), "Cannot determine provider")
  expect_identical(get_provider("o1-preview"), "openai")
  expect_identical(get_provider("o3-mini"), "openai")
  expect_identical(get_provider("o4-mini"), "openai")
})

test_that("built-in provider registry owns detection and processor construction", {
  expect_setequal(
    names(.BUILTIN_PROVIDER_SPECS),
    c(names(.BUILTIN_PROVIDER_PATTERNS), "openrouter")
  )
  expect_silent(validate_builtin_provider_registry())

  processor <- new_builtin_provider_processor(
    "openai",
    "https://proxy.example.test/v1/chat/completions/"
  )
  expect_s3_class(processor, "OpenAIProcessor")
  expect_identical(processor$provider_name, "openai")
  expect_identical(
    processor$base_url,
    "https://proxy.example.test/v1/chat/completions"
  )
  expect_error(
    new_builtin_provider_processor("unsupported"),
    "Unsupported model provider"
  )
})

test_that("get_model_response validates and normalizes prompt text", {
  invalid_prompts <- list(NULL, character(0), c("a", "b"), NA_character_, "   ", 123)
  for (prompt in invalid_prompts) {
    expect_error(
      get_model_response(prompt, "gpt-5.5", "key"),
      "prompt must be a non-empty character scalar"
    )
  }

  result <- testthat::with_mocked_bindings({
    get_model_response("  prompt  ", "gpt-5.5", "key")
  },
  new_builtin_provider_processor = make_mock_builtin_processor_factory(
    function(prompt, model, api_key) {
      expect_identical(prompt, "prompt")
      "ok"
    }
  ))
  expect_identical(result, "ok")
})

test_that("annotate_cell_types validates public scalar controls", {
  base_args <- list(
    input = list("0" = list(genes = c("CD3D"))),
    tissue_name = "PBMC",
    model = "gpt-5.5",
    api_key = NA
  )

  expect_error(
    do.call(annotate_cell_types, modifyList(base_args, list(tissue_name = c("PBMC", "blood")))),
    "tissue_name is required"
  )
  expect_error(
    do.call(annotate_cell_types, modifyList(base_args, list(top_gene_count = 1.5))),
    "top_gene_count must be a positive integer"
  )
  expect_error(
    do.call(annotate_cell_types, modifyList(base_args, list(debug = NA))),
    "debug must be TRUE or FALSE"
  )
})

test_that("annotate_cell_types validates api_key scalar contract", {
  expect_error(
    annotate_cell_types(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      model = "gpt-5.5",
      api_key = c("a", "b")
    ),
    "api_key must be a non-empty character scalar, or NA to return prompt only"
  )
  expect_error(
    annotate_cell_types(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      model = "gpt-5.5",
      api_key = 123
    ),
    "api_key must be a non-empty character scalar, or NA to return prompt only"
  )
})

test_that("interactive_consensus_annotation validates api_keys list contract", {
  expect_error(
    interactive_consensus_annotation(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      models = c("gpt-5.5", "grok-4.3"),
      api_keys = "bad",
      use_cache = FALSE
    ),
    "api_keys must be a named, non-empty list"
  )
  expect_error(
    interactive_consensus_annotation(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      models = c("gpt-5.5", "grok-4.3"),
      api_keys = list(openai = c("first", "second"), grok = "key"),
      use_cache = FALSE
    ),
    "api_keys value for 'openai' must be a character scalar or missing"
  )
})

test_that("model identity normalization trims without rewriting case", {
  expect_identical(
    .normalize_model_vector(c(" MiniMax-M2.7 ", "minimax-m2.7")),
    c("MiniMax-M2.7", "minimax-m2.7")
  )

  cache_manager <- CacheManager$new("temp")
  input <- list("0" = "CD3D")
  expect_false(identical(
    cache_manager$generate_key(input, "MiniMax-M2.7", "0"),
    cache_manager$generate_key(input, "minimax-m2.7", "0")
  ))
})

test_that("multi-model entry points share strict control validation", {
  expect_error(
    compare_model_predictions(
      list("0" = c("G1")),
      "PBMC",
      models = c("gpt-5.5", "gpt-5.5"),
      api_keys = list(openai = "k")
    ),
    "models must not contain duplicate names"
  )
  expect_error(
    compare_model_predictions(
      list("0" = c("G1")),
      "PBMC",
      models = c("gpt-5.5", "grok-4.3"),
      api_keys = list(openai = "k", grok = "k"),
      consensus_threshold = 1.1
    ),
    "consensus_threshold must be a finite number between 0 and 1"
  )

  consensus_args <- list(
    input = list("0" = c("G1")),
    tissue_name = "PBMC",
    models = c("gpt-5.5", "grok-4.3"),
    api_keys = list(openai = "k", grok = "k"),
    use_cache = FALSE
  )
  expect_error(
    do.call(interactive_consensus_annotation, modifyList(consensus_args, list(use_cache = NA))),
    "use_cache must be TRUE or FALSE"
  )
  expect_error(
    do.call(interactive_consensus_annotation, modifyList(consensus_args, list(entropy_threshold = -1))),
    "entropy_threshold must be a finite non-negative number"
  )
  expect_error(
    do.call(interactive_consensus_annotation, modifyList(consensus_args, list(max_discussion_rounds = 1.5))),
    "max_discussion_rounds must be a positive integer"
  )
})

test_that("api key names reject ambiguous normalized duplicates", {
  ambiguous_provider_keys <- list("OpenAI" = "first", " openai " = "second")
  expect_error(
    .normalize_api_keys(ambiguous_provider_keys),
    "api_keys provider names must be unique after case/whitespace normalization"
  )
  expect_error(
    get_api_key(
      "gpt-5.5",
      ambiguous_provider_keys
    ),
    "api_keys provider names must be unique after case/whitespace normalization"
  )
})

test_that("model-specific api key names preserve model identity case", {
  api_keys <- list(
    "MiniMax-M2.7" = "canonical-key",
    "minimax-m2.7" = "different-key"
  )

  expect_identical(get_api_key("MiniMax-M2.7", api_keys), "canonical-key")
  expect_identical(get_api_key("minimax-m2.7", api_keys), "different-key")
  expect_null(get_api_key("MINIMAX-M2.7", api_keys))
})

test_that("get_api_key ignores empty/NA/non-scalar keys and falls back correctly", {
  expect_null(get_api_key("gpt-5.5", list(openai = "")))
  expect_null(get_api_key("gpt-5.5", list(openai = "   ")))
  expect_null(get_api_key("gpt-5.5", list(openai = NA_character_)))
  expect_null(get_api_key("gpt-5.5", list(openai = c("a", "b"))))
  expect_identical(get_api_key("gpt-5.5", list(openai = "", "gpt-5.5" = "k2")), "k2")
  expect_identical(get_api_key("gpt-5.5", list(openai = "  k1  ")), "k1")
  expect_identical(get_api_key("  gpt-5.5  ", list(" OpenAI " = "  k1  ")), "k1")
  expect_null(get_api_key("  gpt-5.5  ", list(" GPT-5.5 " = "  k2  ")))
  expect_identical(get_api_key("  GPT-5.5  ", list(" GPT-5.5 " = "  k2  ")), "k2")
})

test_that("consensus credentials use Gemini environment aliases in priority order", {
  variables <- c("GEMINI_API_KEY", "GOOGLE_API_KEY")
  restore_environment <- preserve_environment_variables(variables)
  on.exit(restore_environment(), add = TRUE)
  Sys.unsetenv(variables)
  Sys.setenv(GOOGLE_API_KEY = "  alias-key  ")

  alias_credential <- .resolve_model_api_key(
    "gemini-3.1-pro-preview",
    api_keys = NULL
  )
  expect_identical(alias_credential$api_key, "alias-key")
  expect_identical(alias_credential$provider, "gemini")
  expect_identical(alias_credential$source, "environment")

  observed_key <- NULL
  result <- testthat::with_mocked_bindings(
    execute_consensus_check(
      "prompt",
      api_keys = NULL,
      models_to_try = "gemini-3.1-pro-preview"
    ),
    get_model_response = function(prompt, model, api_key, base_urls = NULL) {
      observed_key <<- api_key
      "1\n1\n0\nT cell"
    }
  )
  expect_true(result$success)
  expect_identical(observed_key, "alias-key")

  Sys.setenv(GEMINI_API_KEY = "  primary-key  ")
  primary_credential <- .resolve_model_api_key(
    "gemini-3.1-pro-preview",
    api_keys = NULL
  )
  expect_identical(primary_credential$api_key, "primary-key")

  explicit_credential <- .resolve_model_api_key(
    "gemini-3.1-pro-preview",
    api_keys = list(gemini = "  explicit-key  ")
  )
  expect_identical(explicit_credential$api_key, "explicit-key")
  expect_identical(explicit_credential$source, "api_keys")
})

test_that("discussion cache context uses the resolved credential source", {
  variables <- c("GEMINI_API_KEY", "GOOGLE_API_KEY")
  restore_environment <- preserve_environment_variables(variables)
  on.exit(restore_environment(), add = TRUE)
  Sys.unsetenv(variables)
  Sys.setenv(GOOGLE_API_KEY = "alias-key")

  context <- build_discussion_cache_context(
    input = list("0" = "CD3D"),
    cluster_id = "0",
    models = "gemini-3.1-pro-preview",
    api_keys = NULL,
    initial_predictions = list(
      "gemini-3.1-pro-preview" = "Cluster 0: T cell"
    ),
    max_discussion_rounds = 2,
    controversy_threshold = 0.7,
    entropy_threshold = 1,
    consensus_check_model = "gemini-3.1-pro-preview",
    base_urls = list(gemini = "https://proxy.example.test/v1/")
  )

  expect_identical(
    context$requests[[1]][c("provider", "credential_source", "base_url")],
    list(
      provider = "gemini",
      credential_source = "environment",
      base_url = "https://proxy.example.test/v1"
    )
  )
})

test_that("model and api_key are trimmed before R provider dispatch", {
  direct_result <- testthat::with_mocked_bindings({
    get_model_response("prompt", "  gpt-5.5  ", "  key  ")
  },
  new_builtin_provider_processor = make_mock_builtin_processor_factory(
    function(prompt, model, api_key) {
      expect_identical(model, "gpt-5.5")
      expect_identical(api_key, "key")
      "ok"
    }
  ))
  expect_identical(direct_result, "ok")

  public_result <- testthat::with_mocked_bindings({
    suppressMessages(annotate_cell_types(
      input = list("0" = list(genes = c("CD3D"))),
      tissue_name = "PBMC",
      model = "  gpt-5.5  ",
      api_key = "  key  "
    ))
  },
  get_model_response = function(prompt, model, api_key, base_urls = NULL) {
    expect_identical(model, "gpt-5.5")
    expect_identical(api_key, "key")
    c("T cell")
  })
  expect_identical(public_result, c("T cell"))
})

test_that("facilitate_cluster_discussion uses last available consensus on early break", {
  logger_env <- new.env(parent = emptyenv())
  logger_env$events <- list()
  logger <- list(
    log_discussion = function(cluster_id, event_type, data = NULL) {
      logger_env$events <- c(logger_env$events, list(list(event = event_type, data = data)))
    }
  )

  testthat::with_mocked_bindings({
    suppressMessages(facilitate_cluster_discussion(
      cluster_id = "0",
      input = list("0" = list(genes = c("G1"))),
      tissue_name = "PBMC",
      models = c("m1", "m2"),
      api_keys = list(openai = "k"),
      initial_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      top_gene_count = 5,
      max_rounds = 3
    ))
  },
  create_initial_discussion_prompt = function(...) "p1",
  create_discussion_prompt = function(...) "p2",
  get_model_response = local({
    cnt <- 0
    function(...) {
      cnt <<- cnt + 1
      if (cnt <= 2) "CELL TYPE: T cell" else if (cnt == 3) "CELL TYPE: T cell" else "Error: failed"
    }
  }),
  check_consensus = local({
    k <- 0
    function(...) {
      k <<- k + 1
      list(reached = FALSE, consensus_proportion = 0.5, entropy = 1.2, majority_prediction = paste0("M", k))
    }
  }),
  get_api_key = function(...) "k",
  get_provider = function(...) "openai",
  get_logger = function() logger,
  log_warn = function(...) NULL,
  log_info = function(...) NULL)

  end_events <- Filter(function(x) identical(x$event, "end"), logger_env$events)
  expect_equal(length(end_events), 1)
  expect_identical(end_events[[1]]$data$final_result, "M1")
})

test_that("process_controversial_clusters backtracks final prediction when last round lacks consensus", {
  out <- testthat::with_mocked_bindings({
    process_controversial_clusters(
      controversial_clusters = c("0"),
      input = list("0" = list(genes = c("G1"))),
      tissue_name = "PBMC",
      successful_models = c("m1", "m2"),
      api_keys = list(openai = "k"),
      individual_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      top_gene_count = 5,
      controversy_threshold = 0.7,
      entropy_threshold = 1.0,
      max_discussion_rounds = 3,
      cache_manager = list(
        generate_key = function(...) "k",
        save_to_cache = function(...) NULL,
        has_cache = function(...) FALSE
      ),
      use_cache = FALSE
    )
  },
  facilitate_cluster_discussion = function(...) {
    list(
      cluster_id = "0",
      rounds = list(
        list(consensus_result = list(majority_prediction = "T cell")),
        list(responses = list(m1 = "..."))
      )
    )
  },
  log_info = function(...) NULL,
  log_warn = function(...) NULL)

  expect_identical(out$final_annotations[["0"]], "T cell")
})

test_that("process_controversial_clusters ignores invalid cache payloads", {
  discussion_calls <- 0L
  result <- testthat::with_mocked_bindings(
    suppressMessages(process_controversial_clusters(
      controversial_clusters = "0",
      input = list("0" = "G1"),
      tissue_name = "PBMC",
      successful_models = c("m1", "m2"),
      api_keys = list(openai = "key"),
      individual_predictions = list(m1 = "T cell", m2 = "T cell"),
      top_gene_count = 5,
      controversy_threshold = 0.7,
      entropy_threshold = 1,
      max_discussion_rounds = 3,
      cache_manager = list(
        generate_key = function(...) "key",
        load_from_cache = function(...) list(
          annotation = c("T cell", "B cell"),
          discussion_log = list(rounds = list())
        ),
        save_to_cache = function(...) TRUE
      ),
      use_cache = TRUE
    )),
    build_discussion_cache_context = function(...) list(),
    facilitate_cluster_discussion = function(...) {
      discussion_calls <<- discussion_calls + 1L
      list(
        cluster_id = "0",
        rounds = list(list(consensus_result = list(
          majority_prediction = "T cell"
        )))
      )
    },
    log_info = function(...) NULL,
    log_warn = function(...) NULL,
    log_debug = function(...) NULL
  )

  expect_identical(discussion_calls, 1L)
  expect_identical(result$final_annotations[["0"]], "T cell")
})

test_that("check_consensus rejects an LLM indicator that contradicts its metrics", {
  result <- testthat::with_mocked_bindings(
    check_consensus(
      round_responses = c(model_a = "T cell", model_b = "NK cell"),
      api_keys = list(openai = "key"),
      controversy_threshold = 0.7,
      entropy_threshold = 0.8,
      consensus_check_model = "gpt-5.5"
    ),
    execute_consensus_check = function(...) {
      list(success = TRUE, response = "1\n0.5\n1.0\nT cell")
    }
  )

  expect_false(result$reached)
  expect_equal(result$consensus_proportion, 0.5)
  expect_equal(result$entropy, 1)
})

test_that("consensus filtering uses the canonical unknown-annotation contract", {
  invalid_annotations <- c(
    "Unknown",
    "unknown (low confidence)",
    "[Inconclusive]",
    "error: upstream timeout",
    "N/A",
    "--"
  )
  expect_false(any(vapply(
    as.list(invalid_annotations),
    is_real_cell_type_annotation,
    logical(1)
  )))
  expect_true(is_real_cell_type_annotation("CD4+ T cell"))
  expect_identical(clean_annotation("unknown (low confidence)"), "Unknown")

  result <- testthat::with_mocked_bindings(
    suppressMessages(check_consensus(
      round_responses = c(first_model = "unknown", second_model = "UNKNOWN")
    )),
    execute_consensus_check = function(...) {
      stop("unknown annotations must be filtered before an LLM request")
    }
  )

  expect_false(result$reached)
  expect_identical(result$majority_prediction, "Insufficient_Responses")
})

test_that("check_consensus does not expose an invalid LLM majority label", {
  result <- testthat::with_mocked_bindings(
    suppressMessages(check_consensus(
      round_responses = c(first_model = "T cell", second_model = "B cell"),
      api_keys = list(openai = "key"),
      controversy_threshold = 0.5,
      entropy_threshold = 1,
      consensus_check_model = "gpt-5.5"
    )),
    execute_consensus_check = function(...) {
      list(
        success = TRUE,
        response = "1\n0.5\n1\nunknown (low confidence)"
      )
    }
  )

  expect_false(result$reached)
  expect_identical(result$majority_prediction, "T cell")
  expect_equal(result$consensus_proportion, 0.5)
  expect_equal(result$entropy, 1)
})

test_that("check_consensus rejects metrics that cannot arise from the model votes", {
  result <- testthat::with_mocked_bindings(
    suppressMessages(check_consensus(
      round_responses = c(model_a = "T cell", model_b = "NK cell"),
      api_keys = list(openai = "key"),
      controversy_threshold = 0.5,
      entropy_threshold = 1,
      consensus_check_model = "gpt-5.5"
    )),
    execute_consensus_check = function(...) {
      list(success = TRUE, response = "1\n0.9\n0.2\nT cell")
    }
  )

  expect_false(result$reached)
  expect_identical(result$majority_prediction, "T cell")
  expect_equal(result$consensus_proportion, 0.5)
  expect_equal(result$entropy, 1)
})

test_that("a tied vote is not deterministic consensus at a 0.5 threshold", {
  result <- testthat::with_mocked_bindings(
    suppressMessages(check_consensus(
      round_responses = c(first_model = "B cell", second_model = "T cell"),
      api_keys = list(openai = "key"),
      controversy_threshold = 0.5,
      entropy_threshold = 1
    )),
    execute_consensus_check = function(...) {
      list(success = FALSE, response = NULL)
    }
  )

  expect_false(result$reached)
  expect_identical(result$majority_prediction, "B cell")
  expect_equal(result$consensus_proportion, 0.5)
})

test_that("consensus response parsing rejects non-text and ignores blank lines", {
  expect_identical(
    parse_consensus_response(list("1", "1", "0", "T cell")),
    .DEFAULT_CONSENSUS_RESULT
  )
  expect_identical(
    parse_consensus_response("1\n  \n1\n0\nT cell"),
    list(
      reached = TRUE,
      consensus_proportion = 1,
      entropy = 0,
      majority_prediction = "T cell"
    )
  )
})

test_that("flexible consensus parsing requires complete metrics and extracts labels", {
  structured <- parse_consensus_response(paste(
    "Consensus: 1",
    "Consensus Proportion: 0.5",
    "Entropy: 1",
    "Majority Prediction: T cell",
    sep = "\n"
  ))
  expect_identical(structured, list(
    reached = TRUE,
    consensus_proportion = 0.5,
    entropy = 1,
    majority_prediction = "T cell"
  ))

  negative <- parse_consensus_response(c(
    "1",
    "Consensus Proportion = -0.5",
    "Entropy = 1",
    "T cell"
  ))
  oversized <- parse_consensus_response(c(
    "1",
    "Consensus Proportion = 1.5",
    "Entropy = 1",
    "T cell"
  ))
  expect_identical(negative$consensus_proportion, 0)
  expect_identical(oversized$consensus_proportion, 0)
})

test_that("prepare_models_list only adds valid direct-provider fallbacks", {
  openai_models <- prepare_models_list("openai/gpt-5.5")
  unsupported_models <- prepare_models_list("meta-llama/llama-3.3-70b")

  expect_identical(openai_models[1:2], c("openai/gpt-5.5", "gpt-5.5"))
  expect_identical(unsupported_models[[1]], "meta-llama/llama-3.3-70b")
  expect_false("llama-3.3-70b" %in% unsupported_models)
})

test_that("model requests retry only transient API failures", {
  run_failure <- function(status_code) {
    state <- new.env(parent = emptyenv())
    state$calls <- 0L
    state$waits <- numeric(0)
    error <- tryCatch(testthat::with_mocked_bindings(
      get_model_response("prompt", "gpt-5.5", "key"),
      new_builtin_provider_processor = make_mock_builtin_processor_factory(
        function(...) {
          state$calls <- state$calls + 1L
          stop_api_request_error("request failed", status_code = status_code)
        }
      ),
      wait_before_model_retry = function(seconds) {
        state$waits <- c(state$waits, seconds)
      }
    ), error = identity)
    list(error = error, calls = state$calls, waits = state$waits)
  }

  permanent <- run_failure(401L)
  transient <- run_failure(429L)

  expect_s3_class(permanent$error, "mllm_api_error")
  expect_identical(permanent$calls, 1L)
  expect_length(permanent$waits, 0)
  expect_s3_class(transient$error, "mllm_api_error")
  expect_identical(transient$calls, 3L)
  expect_identical(transient$waits, c(5, 10))
})

test_that("model request retry returns the first successful transient recovery", {
  calls <- 0L
  waits <- numeric(0)

  result <- testthat::with_mocked_bindings(
    get_model_response("prompt", "gpt-5.5", "key"),
    new_builtin_provider_processor = make_mock_builtin_processor_factory(
      function(...) {
        calls <<- calls + 1L
        if (calls == 1L) {
          stop_api_request_error("rate limited", status_code = 429L)
        }
        "T cell"
      }
    ),
    wait_before_model_retry = function(seconds) {
      waits <<- c(waits, seconds)
    }
  )

  expect_identical(result, "T cell")
  expect_identical(calls, 2L)
  expect_identical(waits, 5)
})

test_that("consensus candidate orchestration does not add another retry layer", {
  calls <- 0L
  result <- suppressWarnings(testthat::with_mocked_bindings(
    execute_consensus_check(
      "prompt",
      list(openai = "key"),
      "gpt-5.5"
    ),
    get_model_response = function(...) {
      calls <<- calls + 1L
      stop_api_request_error("request failed", status_code = 429L)
    }
  ))

  expect_false(result$success)
  expect_identical(calls, 1L)
})

test_that("facilitate_cluster_discussion logs consensus and end on first-round early return", {
  logger_env <- new.env(parent = emptyenv())
  logger_env$events <- list()
  logger <- list(
    log_discussion = function(cluster_id, event_type, data = NULL) {
      logger_env$events <- c(logger_env$events, list(list(event = event_type, data = data)))
    }
  )

  testthat::with_mocked_bindings({
    facilitate_cluster_discussion(
      cluster_id = "0",
      input = list("0" = list(genes = c("G1"))),
      tissue_name = "PBMC",
      models = c("m1", "m2"),
      api_keys = list(openai = "k"),
      initial_predictions = list(m1 = c("0:T"), m2 = c("0:T")),
      top_gene_count = 5,
      max_rounds = 3
    )
  },
  create_initial_discussion_prompt = function(...) "p1",
  create_discussion_prompt = function(...) "p2",
  get_model_response = function(...) "Error: failed",
  check_consensus = function(...) list(reached = FALSE, consensus_proportion = 0, entropy = 0, majority_prediction = "Unknown"),
  get_api_key = function(...) "k",
  get_provider = function(...) "openai",
  get_logger = function() logger,
  log_warn = function(...) NULL,
  log_info = function(...) NULL)

  event_names <- vapply(logger_env$events, function(x) x$event, character(1))
  expect_true("consensus" %in% event_names)
  expect_true("end" %in% event_names)
})

test_that("BaseAPIProcessor marks semantic error result as failed API call", {
  logger_env <- new.env(parent = emptyenv())
  logger_env$api_call_success <- logical(0)
  logger_env$failed_audit_logs <- 0L

  logger <- list(
    info = function(...) NULL,
    debug = function(...) NULL,
    error = function(...) NULL,
    log_api_call = function(provider, model, duration, success = TRUE, tokens = NULL) {
      logger_env$api_call_success <- c(logger_env$api_call_success, success)
    },
    log_api_request_response = function(...) {
      logger_env$failed_audit_logs <- logger_env$failed_audit_logs + 1L
    }
  )

  testthat::with_mocked_bindings({
    TestProcessor <- R6::R6Class("TestProcessor",
      inherit = BaseAPIProcessor,
      public = list(
        initialize = function() super$initialize("testp", NULL),
        get_default_api_url = function() "https://example.test",
        make_api_call = function(chunk_content, model, api_key) stop("simulated failure"),
        extract_response_content = function(response, model) "unused"
      )
    )

    proc <- TestProcessor$new()
    expect_error(proc$process_request("prompt", "test-model", "test-key"),
                 "simulated failure")
  },
  get_logger = function() logger)

  expect_true(length(logger_env$api_call_success) >= 1)
  expect_identical(tail(logger_env$api_call_success, 1), FALSE)
  # Failed requests should still produce audit logs for traceability
  expect_true(logger_env$failed_audit_logs >= 1L)
})

test_that("initialize_logger preserves logger configuration and regenerates unique session IDs", {
  initialize_logger("logs")
  configure_logger(level = "DEBUG", console_output = FALSE, json_format = FALSE)
  logger_cfg <- get_logger()
  logger_cfg$max_log_size <- 42
  logger_cfg$max_log_files <- 7

  initialize_logger("logs")
  logger_after_reinit <- get_logger()
  expect_identical(logger_after_reinit$log_level, "DEBUG")
  expect_identical(logger_after_reinit$enable_console, FALSE)
  expect_identical(logger_after_reinit$enable_json, FALSE)
  expect_identical(logger_after_reinit$max_log_size, 42)
  expect_equal(logger_after_reinit$max_log_files, 7)

  id1 <- logger_after_reinit$session_id
  initialize_logger("logs")
  id2 <- get_logger()$session_id
  expect_false(identical(id1, id2))
})

test_that("UnifiedLogger validates configuration before mutating state", {
  log_dir <- tempfile("mllm-logger-")
  logger <- UnifiedLogger$new(base_dir = log_dir, level = "INFO")

  expect_error(logger$set_level("verbose"), "level must be one of")
  expect_identical(logger$log_level, "INFO")
  expect_error(
    UnifiedLogger$new(base_dir = log_dir, max_files = 0),
    "max_files must be a positive integer"
  )
  expect_error(
    UnifiedLogger$new(base_dir = log_dir, console_output = NA),
    "console_output must be TRUE or FALSE"
  )
})

test_that("UnifiedLogger counts cache misses once and sanitizes discussion paths", {
  restore_check_environment <- enable_logger_file_writes()
  on.exit(restore_check_environment(), add = TRUE)
  log_dir <- tempfile("mllm-logger-")
  logger <- UnifiedLogger$new(base_dir = log_dir)

  logger$log_cache_operation("miss", "key")
  expect_identical(logger$performance_stats$cache_misses, 1)

  logger$log_discussion(
    "../../outside",
    "start",
    list(tissue_name = "PBMC", marker_genes = "G1")
  )
  created_files <- list.files(log_dir, recursive = TRUE, full.names = TRUE)
  expect_true(any(grepl("cluster_.*outside_discussion\\.md$", created_files)))
  expect_true(all(startsWith(normalizePath(created_files), normalizePath(log_dir))))
})

test_that("UnifiedLogger force cleanup is stable for fewer than max files", {
  restore_check_environment <- enable_logger_file_writes()
  on.exit(restore_check_environment(), add = TRUE)
  log_dir <- tempfile("mllm-logger-")
  logger <- UnifiedLogger$new(base_dir = log_dir, max_files = 5)
  expect_true(length(list.files(log_dir, pattern = "^mllm_.*\\.log$")) >= 1)

  expect_no_error(logger$cleanup_logs(force = TRUE))
  expect_length(list.files(log_dir, pattern = "^mllm_.*\\.log$"), 0)
})

test_that("UnifiedLogger file failures do not change business control flow", {
  restore_check_environment <- enable_logger_file_writes()
  on.exit(restore_check_environment(), add = TRUE)
  log_dir <- tempfile("mllm-logger-")
  logger <- UnifiedLogger$new(base_dir = log_dir)
  blocked_path <- tempfile("mllm-log-file-")
  writeLines("not a directory", blocked_path)
  logger$log_dir <- blocked_path

  expect_no_error(logger$info("non-fatal"))
  expect_null(logger$log_model_response("openai", "gpt-5.5", "T cell"))
  expect_null(logger$log_api_request_response(
    "../provider",
    "gpt-5.5",
    "prompt",
    "T cell"
  ))
  expect_null(logger$log_discussion(
    "../../cluster",
    "start",
    list(tissue_name = "PBMC", marker_genes = "G1")
  ))
})
