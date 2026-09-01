# Grok API Processor

Grok API Processor

Grok API Processor

## Details

Concrete implementation of BaseAPIProcessor for Grok models. Handles
Grok-specific API calls, authentication, and response parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `GrokProcessor`

## Methods

### Public methods

- [`GrokProcessor$new()`](#method-GrokProcessor-new)

- [`GrokProcessor$get_default_api_url()`](#method-GrokProcessor-get_default_api_url)

- [`GrokProcessor$make_api_call()`](#method-GrokProcessor-make_api_call)

- [`GrokProcessor$extract_response_content()`](#method-GrokProcessor-extract_response_content)

- [`GrokProcessor$clone()`](#method-GrokProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$extract_usage()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-extract_usage)
- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize Grok processor

#### Usage

    GrokProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default Grok API URL

#### Usage

    GrokProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to Grok

#### Usage

    GrokProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier

- `api_key`:

  Grok API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Grok API response

#### Usage

    GrokProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    GrokProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
