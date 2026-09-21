# Tests for the LiteLLM gateway provider

test_that("the provider is selected explicitly, not inferred from the model", {
  # LiteLLM routes any vendor's model, so it carries no name pattern and is
  # chosen by name, exactly like the openrouter provider.
  spec <- get_builtin_provider_spec("litellm")

  expect_equal(spec$processor_class, "LiteLLMProcessor")
  expect_equal(spec$display_name, "LiteLLM")
  expect_null(spec$pattern)
})

test_that("adding the provider leaves model-name inference unchanged", {
  expect_equal(get_provider("gpt-5.5"), "openai")
  expect_equal(get_provider("claude-opus-4-7"), "anthropic")
  expect_equal(get_provider("anthropic/claude-sonnet-4.6"), "openrouter")
  expect_equal(get_provider("openai/gpt-5.5"), "openrouter")
})

test_that("the built-in provider registry stays valid with LiteLLM added", {
  expect_silent(validate_builtin_provider_registry())
})

test_that("LiteLLMProcessor is resolvable and defaults to a local gateway", {
  processor <- new_builtin_provider_processor("litellm")

  expect_s3_class(processor, "LiteLLMProcessor")
  expect_equal(processor$get_default_api_url(), "http://localhost:4000/v1/chat/completions")
  expect_equal(get_builtin_provider_display_name("litellm"), "LiteLLM")
})

test_that("a custom gateway endpoint overrides the default", {
  processor <- new_builtin_provider_processor(
    "litellm",
    base_url = "https://gw.example.com/v1/chat/completions"
  )

  expect_equal(processor$get_api_url(), "https://gw.example.com/v1/chat/completions")
})

test_that("the processor implements the required interface", {
  processor <- new_builtin_provider_processor("litellm")

  for (method in c("get_default_api_url", "make_api_call", "extract_response_content")) {
    expect_true(is.function(processor[[method]]))
  }
})

test_that("the gateway spec carries no api key alias, since the key is optional", {
  # A gateway started without a master key serves unauthenticated requests.
  spec <- get_builtin_provider_spec("litellm")

  expect_null(spec$api_key_env_aliases)
})
