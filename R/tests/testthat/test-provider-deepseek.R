# Tests for the DeepSeek provider

make_deepseek_json_response <- function(payload, status_code = 200L) {
  structure(
    list(
      url = "https://api.deepseek.com/v1/chat/completions",
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

quiet_deepseek_logger <- function() {
  list(
    info = function(...) NULL,
    debug = function(...) NULL,
    warn = function(...) NULL,
    error = function(...) NULL,
    log_api_call = function(...) NULL,
    log_api_request_response = function(...) NULL
  )
}

test_that("DeepSeekProcessor disables thinking for annotation requests", {
  captured <- new.env(parent = emptyenv())

  testthat::with_mocked_bindings({
    testthat::with_mocked_bindings({
      processor <- DeepSeekProcessor$new()
      response <- processor$make_api_call(
        "genes",
        "deepseek-v4-flash",
        "test-key"
      )

      expect_identical(captured$url, "https://api.deepseek.com/v1/chat/completions")
      expect_identical(captured$headers[["Authorization"]], "Bearer test-key")
      expect_identical(captured$body$model, "deepseek-v4-flash")
      expect_identical(
        captured$body$messages,
        list(list(role = "user", content = "genes"))
      )
      expect_identical(captured$body$temperature, 0.7)
      expect_identical(captured$body$max_tokens, 4096)
      expect_false(captured$body$stream)
      expect_identical(captured$body$thinking, list(type = "disabled"))
      expect_s3_class(response, "response")
    },
    POST = function(url, config = NULL, body = NULL, ...) {
      captured$url <- url
      captured$headers <- config$headers
      captured$body <- body
      make_deepseek_json_response(list(
        choices = list(list(message = list(content = "Cluster 1: T cells")))
      ))
    },
    .package = "httr")
  },
  get_logger = quiet_deepseek_logger)
})
