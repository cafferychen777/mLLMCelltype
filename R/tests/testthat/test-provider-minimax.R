# Tests for MiniMax in-body business-error retry classification.
#
# MiniMax reports transient failures (rate limits, timeouts) via HTTP 200 with a
# non-zero base_resp$status_code, bypassing HTTP-status retry classification.
# These must surface as retryable mllm_api_error conditions so the shared retry
# loop backs off instead of dropping the model from the consensus panel.

test_that("MiniMax transient business errors are retryable, permanent ones are not", {
  quiet_logger <- list(
    info = function(...) NULL,
    debug = function(...) NULL,
    warn = function(...) NULL,
    error = function(...) NULL,
    log_api_call = function(...) NULL
  )

  classify <- function(status_code) {
    testthat::with_mocked_bindings(
      {
        processor <- MinimaxProcessor$new()
        testthat::with_mocked_bindings(
          tryCatch(
            processor$extract_response_content(
              structure(list(), class = "response"),
              "MiniMax-M2.7"
            ),
            error = identity
          ),
          content = function(...) {
            list(base_resp = list(status_code = status_code, status_msg = "boom"))
          },
          .package = "httr"
        )
      },
      get_logger = function() quiet_logger
    )
  }

  # 1002 (rate limit) and 1001 (timeout) are transient -> retryable.
  rate_limited <- classify(1002L)
  expect_s3_class(rate_limited, "mllm_api_error")
  expect_true(rate_limited$retryable)

  timed_out <- classify(1001L)
  expect_true(timed_out$retryable)

  # 1004 (auth) and 2013 (invalid params) are fatal -> not retryable.
  unauthorized <- classify(1004L)
  expect_s3_class(unauthorized, "mllm_api_error")
  expect_false(unauthorized$retryable)

  invalid_params <- classify(2013L)
  expect_false(invalid_params$retryable)
})

test_that("get_model_response retries a transient MiniMax business error end-to-end", {
  # Drives the real retry loop: two transient (1002) responses then a success.
  quiet_logger <- list(
    info = function(...) NULL, debug = function(...) NULL,
    warn = function(...) NULL, error = function(...) NULL,
    log_api_call = function(...) NULL,
    log_api_request_response = function(...) NULL
  )
  state <- new.env(parent = emptyenv())
  state$attempt <- 0L

  result <- testthat::with_mocked_bindings(
    testthat::with_mocked_bindings(
      get_model_response("genes", "MiniMax-M2.7", "test-key"),
      POST = function(...) {
        state$attempt <- state$attempt + 1L
        structure(list(), class = "response")
      },
      status_code = function(...) 200L,
      content = function(...) {
        if (state$attempt < 3L) {
          list(base_resp = list(status_code = 1002L, status_msg = "rate limit"))
        } else {
          list(choices = list(list(message = list(content = "Cluster 1: T cells"))))
        }
      },
      .package = "httr"
    ),
    get_logger = function() quiet_logger,
    wait_before_model_retry = function(seconds) invisible(NULL)
  )

  expect_identical(state$attempt, 3L)
  expect_identical(result, "Cluster 1: T cells")
})
