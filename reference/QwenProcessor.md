# Qwen API Processor

Qwen API Processor

Qwen API Processor

## Details

Concrete implementation of BaseAPIProcessor for Qwen models. Handles
Qwen-specific API calls, authentication, and response parsing.

Qwen has two API endpoints:

- International:
  https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation
  (preferred)

- Domestic (China):
  https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
  (fallback) The processor automatically tries international first, then
  falls back to domestic if needed.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
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

- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Test if an endpoint is accessible

Initialize Qwen processor

#### Usage

    QwenProcessor$new(base_url = NULL)

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default Qwen API URL with intelligent endpoint selection

#### Usage

    QwenProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `get_working_api_url()`

Get working Qwen API URL with automatic endpoint detection

#### Usage

    QwenProcessor$get_working_api_url(api_key)

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to Qwen

#### Usage

    QwenProcessor$make_api_call(chunk_content, model, api_key)

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Qwen API response

#### Usage

    QwenProcessor$extract_response_content(response, model)

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    QwenProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
