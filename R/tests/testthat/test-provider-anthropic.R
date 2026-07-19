# Tests for the Anthropic multi-block response parser.
#
# Anthropic Messages responses can contain multiple content blocks (e.g. a
# leading thinking/reasoning block). Hard-indexing the first block would discard
# the answer, so the parser iterates and concatenates the text blocks.

test_that("Anthropic extract_response_content concatenates text and skips thinking blocks", {
  quiet_logger <- list(
    info = function(...) NULL,
    debug = function(...) NULL,
    warn = function(...) NULL,
    error = function(...) NULL,
    log_api_call = function(...) NULL
  )

  extract <- function(blocks) {
    testthat::with_mocked_bindings(
      {
        processor <- AnthropicProcessor$new()
        testthat::with_mocked_bindings(
          processor$extract_response_content(
            structure(list(), class = "response"),
            "claude-opus-4-7"
          ),
          content = function(...) list(content = blocks),
          .package = "httr"
        )
      },
      get_logger = function() quiet_logger
    )
  }

  # A leading thinking block must not discard the answer.
  answer <- extract(list(
    list(type = "thinking", thinking = "reasoning..."),
    list(type = "text", text = "Cluster 1: T cells")
  ))
  expect_identical(answer, "Cluster 1: T cells")

  # Multiple text blocks are concatenated in order.
  multi <- extract(list(
    list(type = "text", text = "Cluster 1: T cells"),
    list(type = "text", text = "Cluster 2: B cells")
  ))
  expect_identical(multi, "Cluster 1: T cells\nCluster 2: B cells")

  # A response with no text block is rejected.
  err <- tryCatch(
    extract(list(list(type = "thinking", thinking = "reasoning..."))),
    error = identity
  )
  expect_s3_class(err, "error")
  expect_match(conditionMessage(err), "Unexpected response format from Anthropic")
})

test_that("Anthropic parser warns when the response is truncated at max_tokens", {
  warnings_seen <- character(0)
  recording_logger <- list(
    info = function(...) NULL,
    debug = function(...) NULL,
    warn = function(message, context = NULL) {
      warnings_seen <<- c(warnings_seen, message)
      invisible(NULL)
    },
    error = function(...) NULL,
    log_api_call = function(...) NULL
  )

  result <- testthat::with_mocked_bindings(
    {
      processor <- AnthropicProcessor$new()
      testthat::with_mocked_bindings(
        processor$extract_response_content(
          structure(list(), class = "response"),
          "claude-opus-4-7"
        ),
        content = function(...) {
          list(
            content = list(list(type = "text", text = "Cluster 1: T cells")),
            stop_reason = "max_tokens"
          )
        },
        .package = "httr"
      )
    },
    get_logger = function() recording_logger
  )

  expect_identical(result, "Cluster 1: T cells")
  expect_true(any(grepl("truncated", warnings_seen)))
})

test_that("Anthropic extraction tolerates a non-list content block", {
  quiet_logger <- list(
    info = function(...) NULL, debug = function(...) NULL,
    warn = function(...) NULL, error = function(...) NULL,
    log_api_call = function(...) NULL
  )
  result <- testthat::with_mocked_bindings(
    {
      processor <- AnthropicProcessor$new()
      testthat::with_mocked_bindings(
        processor$extract_response_content(
          structure(list(), class = "response"), "claude-opus-4-7"
        ),
        content = function(...) {
          list(content = list("bare string", list(type = "text", text = "Cluster 1: T cells")))
        },
        .package = "httr"
      )
    },
    get_logger = function() quiet_logger
  )
  expect_identical(result, "Cluster 1: T cells")
})

test_that("Anthropic parser stays silent on a normal stop_reason", {
  warnings_seen <- character(0)
  recording_logger <- list(
    info = function(...) NULL, debug = function(...) NULL,
    warn = function(message, context = NULL) {
      warnings_seen <<- c(warnings_seen, message)
      invisible(NULL)
    },
    error = function(...) NULL, log_api_call = function(...) NULL
  )
  testthat::with_mocked_bindings(
    {
      processor <- AnthropicProcessor$new()
      testthat::with_mocked_bindings(
        processor$extract_response_content(
          structure(list(), class = "response"), "claude-opus-4-7"
        ),
        content = function(...) {
          list(
            content = list(list(type = "text", text = "Cluster 1: T cells")),
            stop_reason = "end_turn"
          )
        },
        .package = "httr"
      )
    },
    get_logger = function() recording_logger
  )
  expect_false(any(grepl("truncated", warnings_seen)))
})
