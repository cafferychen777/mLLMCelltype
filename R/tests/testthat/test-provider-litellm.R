# Tests for the LiteLLM gateway provider

test_that("LiteLLM models route to the gateway, not OpenRouter", {
  # get_provider() returns "openrouter" for ANY model containing '/'. That rule
  # runs before prefix matching, so without an explicit slash-prefix pass first
  # every litellm/* model silently routes to OpenRouter and fails there.
  expect_equal(get_provider("litellm/gpt-5.5"), "litellm")
  expect_equal(get_provider("litellm/claude-opus-4-7"), "litellm")
})

test_that("a nested gateway alias still routes to the gateway", {
  expect_equal(get_provider("litellm/anthropic/claude-opus-4-7"), "litellm")
})

test_that("routing is case insensitive", {
  expect_equal(get_provider("LiteLLM/GPT-5.5"), "litellm")
})

test_that("OpenRouter still claims other namespaced models", {
  expect_equal(get_provider("anthropic/claude-sonnet-4.6"), "openrouter")
  expect_equal(get_provider("openai/gpt-5.5"), "openrouter")
})

test_that("bare model names are unaffected", {
  expect_equal(get_provider("gpt-5.5"), "openai")
  expect_equal(get_provider("claude-opus-4-7"), "anthropic")
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

test_that("the routing prefix is stripped before the request is built", {
  # The gateway knows nothing about the litellm/ prefix; it exists only so
  # mLLMCelltype can pick this processor.
  expect_equal(strip_litellm_model_prefix("litellm/gpt-5.5"), "gpt-5.5")
  expect_equal(strip_litellm_model_prefix("LiteLLM/claude-opus-4-7"), "claude-opus-4-7")
  expect_equal(
    strip_litellm_model_prefix("litellm/anthropic/claude-opus-4-7"),
    "anthropic/claude-opus-4-7"
  )
})

test_that("an unprefixed model name passes through unchanged", {
  expect_equal(strip_litellm_model_prefix("gpt-5.5"), "gpt-5.5")
})

test_that("the processor implements the required interface", {
  processor <- new_builtin_provider_processor("litellm")

  for (method in c("get_default_api_url", "make_api_call", "extract_response_content")) {
    expect_true(is.function(processor[[method]]))
  }
})

test_that("the gateway spec carries no api key alias, since the key is optional", {
  spec <- get_builtin_provider_spec("litellm")

  expect_equal(spec$processor_class, "LiteLLMProcessor")
  expect_equal(spec$pattern, "^litellm/")
  expect_null(spec$api_key_env_aliases)
})
