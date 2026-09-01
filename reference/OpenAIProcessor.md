# OpenAI API Processor

OpenAI API Processor

OpenAI API Processor

## Details

Concrete implementation of BaseAPIProcessor for OpenAI models. Handles
OpenAI-specific API calls, authentication, and response parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `OpenAIProcessor`

## Methods

### Public methods

- [`OpenAIProcessor$new()`](#method-OpenAIProcessor-new)

- [`OpenAIProcessor$get_default_api_url()`](#method-OpenAIProcessor-get_default_api_url)

- [`OpenAIProcessor$make_api_call()`](#method-OpenAIProcessor-make_api_call)

- [`OpenAIProcessor$extract_response_content()`](#method-OpenAIProcessor-extract_response_content)

- [`OpenAIProcessor$clone()`](#method-OpenAIProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$extract_usage()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-extract_usage)
- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize OpenAI processor

#### Usage

    OpenAIProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default OpenAI API URL

#### Usage

    OpenAIProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to OpenAI

#### Usage

    OpenAIProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier

- `api_key`:

  OpenAI API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from OpenAI API response

#### Usage

    OpenAIProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    OpenAIProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
