# DeepSeek API Processor

DeepSeek API Processor

DeepSeek API Processor

## Details

Concrete implementation of BaseAPIProcessor for DeepSeek models. Handles
DeepSeek-specific API calls, authentication, and response parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `DeepSeekProcessor`

## Methods

### Public methods

- [`DeepSeekProcessor$new()`](#method-DeepSeekProcessor-new)

- [`DeepSeekProcessor$get_default_api_url()`](#method-DeepSeekProcessor-get_default_api_url)

- [`DeepSeekProcessor$make_api_call()`](#method-DeepSeekProcessor-make_api_call)

- [`DeepSeekProcessor$extract_response_content()`](#method-DeepSeekProcessor-extract_response_content)

- [`DeepSeekProcessor$clone()`](#method-DeepSeekProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize DeepSeek processor

#### Usage

    DeepSeekProcessor$new(base_url = NULL)

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default DeepSeek API URL

#### Usage

    DeepSeekProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to DeepSeek

#### Usage

    DeepSeekProcessor$make_api_call(chunk_content, model, api_key)

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from DeepSeek API response

#### Usage

    DeepSeekProcessor$extract_response_content(response, model)

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    DeepSeekProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
