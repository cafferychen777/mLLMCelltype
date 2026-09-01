# OpenRouter API Processor

OpenRouter API Processor

OpenRouter API Processor

## Details

Concrete implementation of BaseAPIProcessor for OpenRouter models.
Handles OpenRouter-specific API calls, authentication, and response
parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `OpenRouterProcessor`

## Methods

### Public methods

- [`OpenRouterProcessor$new()`](#method-OpenRouterProcessor-new)

- [`OpenRouterProcessor$get_default_api_url()`](#method-OpenRouterProcessor-get_default_api_url)

- [`OpenRouterProcessor$make_api_call()`](#method-OpenRouterProcessor-make_api_call)

- [`OpenRouterProcessor$extract_response_content()`](#method-OpenRouterProcessor-extract_response_content)

- [`OpenRouterProcessor$clone()`](#method-OpenRouterProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$extract_usage()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-extract_usage)
- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize OpenRouter processor

#### Usage

    OpenRouterProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default OpenRouter API URL

#### Usage

    OpenRouterProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to OpenRouter

#### Usage

    OpenRouterProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier

- `api_key`:

  OpenRouter API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from OpenRouter API response

#### Usage

    OpenRouterProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    OpenRouterProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
