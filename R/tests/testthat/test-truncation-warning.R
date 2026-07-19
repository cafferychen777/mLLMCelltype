# Tests for the shared OpenAI-compatible chat-completions truncation warning.
#
# When a provider stops because it hit max_tokens (finish_reason == "length"),
# the trailing clusters are silently dropped; the extractor must surface a
# warning so this is diagnosable, and must stay silent on a normal stop.

make_recording_logger <- function(sink) {
  list(
    info = function(...) NULL,
    debug = function(...) NULL,
    warn = function(message, context = NULL) {
      sink$messages <- c(sink$messages, message)
      invisible(NULL)
    },
    error = function(...) NULL,
    log_api_call = function(...) NULL
  )
}

test_that("chat-completions extraction warns when finish_reason is 'length'", {
  sink <- new.env()
  sink$messages <- character(0)
  logger <- make_recording_logger(sink)

  extract <- function(finish_reason) {
    testthat::with_mocked_bindings(
      {
        processor <- OpenAIProcessor$new()
        testthat::with_mocked_bindings(
          processor$extract_response_content(
            structure(list(), class = "response"), "gpt-5.5"
          ),
          content = function(...) {
            list(choices = list(list(
              message = list(content = "Cluster 1: T cells"),
              finish_reason = finish_reason
            )))
          },
          .package = "httr"
        )
      },
      get_logger = function() logger
    )
  }

  result <- extract("length")
  expect_identical(result, "Cluster 1: T cells")
  expect_true(any(grepl("truncated", sink$messages)))

  sink$messages <- character(0)
  extract("stop")
  expect_false(any(grepl("truncated", sink$messages)))
})
