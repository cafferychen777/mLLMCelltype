test_that("API request timeout defaults to 120 seconds", {
  old_option <- options(mLLMCelltype.api_timeout = NULL)
  on.exit(options(old_option), add = TRUE)

  expect_identical(get_api_request_timeout(), 120)
})

test_that("API request timeout accepts a positive numeric override", {
  old_option <- options(mLLMCelltype.api_timeout = 300)
  on.exit(options(old_option), add = TRUE)

  expect_identical(get_api_request_timeout(), 300)
})

test_that("API request timeout rejects invalid overrides", {
  old_option <- options(mLLMCelltype.api_timeout = 0)
  on.exit(options(old_option), add = TRUE)

  expect_error(
    get_api_request_timeout(),
    "must be one positive number of seconds",
    fixed = TRUE
  )
})
