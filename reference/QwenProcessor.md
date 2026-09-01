# Qwen API Processor

Qwen API Processor

Qwen API Processor

## Details

Concrete implementation of BaseAPIProcessor for Qwen models. Handles
Qwen-specific API calls, authentication, and response parsing.

Qwen has OpenAI-compatible chat completions endpoints:

- International (US):
  https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions

- Domestic (China):
  https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

- Legacy international:
  https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions
  The processor automatically tries international first, then domestic,
  then legacy international.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `QwenProcessor`

## Methods

### Public methods

- [`QwenProcessor$new()`](#method-QwenProcessor-new)

- [`QwenProcessor$get_default_api_url()`](#method-QwenProcessor-get_default_api_url)

- [`QwenProcessor$get_working_api_url()`](#method-QwenProcessor-get_working_api_url)

- [`QwenProcessor$make_api_call()`](#method-QwenProcessor-make_api_call)

- [`QwenProcessor$extract_response_content()`](#method-QwenProcessor-extract_response_content)

- [`QwenProcessor$clone()`](#method-QwenProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$extract_usage()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-extract_usage)
- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Test if an endpoint is accessible

Initialize Qwen processor

#### Usage

    QwenProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default Qwen OpenAI-compatible chat completions API URL

#### Usage

    QwenProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `get_working_api_url()`

Get working Qwen API URL with automatic endpoint detection

#### Usage

    QwenProcessor$get_working_api_url(api_key)

#### Arguments

- `api_key`:

  Qwen API key used for regional endpoint probing

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to Qwen

#### Usage

    QwenProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier

- `api_key`:

  Qwen API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Qwen API response

#### Usage

    QwenProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    QwenProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
