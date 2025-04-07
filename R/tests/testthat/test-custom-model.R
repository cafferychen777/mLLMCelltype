test_that("custom provider registration works", {
  # Test registering a valid provider
  expect_true(register_custom_provider(
    provider_name = "test_provider",
    process_fn = function(prompt, model, api_key) {
      return("test response")
    }
  ))
  
  # Test registering with invalid arguments
  expect_error(register_custom_provider(
    provider_name = 123,
    process_fn = function() {}
  ))
  
  # Test registering with missing required arguments
  expect_error(register_custom_provider(
    provider_name = "test_provider2",
    process_fn = function(prompt) {}
  ))
  
  # Test duplicate provider registration
  expect_error(register_custom_provider(
    provider_name = "test_provider",
    process_fn = function(prompt, model, api_key) {}
  ))
})

test_that("custom model registration works", {
  # Register a provider first
  register_custom_provider(
    provider_name = "model_test_provider",
    process_fn = function(prompt, model, api_key) {
      return("test response")
    }
  )
  
  # Test registering a valid model
  expect_true(register_custom_model(
    model_name = "test_model",
    provider_name = "model_test_provider",
    model_config = list(
      temperature = 0.7,
      max_tokens = 2000
    )
  ))
  
  # Test registering with invalid provider
  expect_error(register_custom_model(
    model_name = "test_model2",
    provider_name = "nonexistent_provider"
  ))
  
  # Test duplicate model registration
  expect_error(register_custom_model(
    model_name = "test_model",
    provider_name = "model_test_provider"
  ))
})

test_that("get_provider works with custom models", {
  # Register a provider and model
  register_custom_provider(
    provider_name = "provider_test",
    process_fn = function(prompt, model, api_key) {
      return("test response")
    }
  )
  
  register_custom_model(
    model_name = "provider_test_model",
    provider_name = "provider_test"
  )
  
  # Test get_provider with custom model
  expect_equal(get_provider("provider_test_model"), "provider_test")
  
  # Test get_provider with non-existent model
  expect_error(get_provider("nonexistent_model"))
})

test_that("custom model processing works", {
  # Register a provider with a specific response
  register_custom_provider(
    provider_name = "process_test_provider",
    process_fn = function(prompt, model, api_key) {
      return("Expected test response")
    }
  )
  
  register_custom_model(
    model_name = "process_test_model",
    provider_name = "process_test_provider",
    model_config = list(temperature = 0.7)
  )
  
  # Test processing with custom model
  expect_equal(
    process_custom("Test prompt", "process_test_model", "test_key"),
    "Expected test response"
  )
  
  # Test processing with non-existent model
  expect_error(
    process_custom("Test prompt", "nonexistent_model", "test_key")
  )
})
