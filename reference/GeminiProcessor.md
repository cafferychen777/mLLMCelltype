# Gemini API Processor

Gemini API Processor

Gemini API Processor

## Details

Concrete implementation of BaseAPIProcessor for Gemini models. Handles
Gemini-specific API calls, authentication, and response parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `GeminiProcessor`

## Methods

### Public methods

- [`GeminiProcessor$new()`](#method-GeminiProcessor-new)

- [`GeminiProcessor$get_default_api_url()`](#method-GeminiProcessor-get_default_api_url)

- [`GeminiProcessor$get_api_url_for_model()`](#method-GeminiProcessor-get_api_url_for_model)

- [`GeminiProcessor$make_api_call()`](#method-GeminiProcessor-make_api_call)

- [`GeminiProcessor$extract_response_content()`](#method-GeminiProcessor-extract_response_content)

- [`GeminiProcessor$extract_usage()`](#method-GeminiProcessor-extract_usage)

- [`GeminiProcessor$clone()`](#method-GeminiProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize Gemini processor

#### Usage

    GeminiProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default Gemini API URL template

#### Usage

    GeminiProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `get_api_url_for_model()`

Get API URL for specific model

#### Usage

    GeminiProcessor$get_api_url_for_model(model)

#### Arguments

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to Gemini

#### Usage

    GeminiProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier

- `api_key`:

  Gemini API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Gemini API response

#### Usage

    GeminiProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `extract_usage()`

Extract normalized Gemini token usage

#### Usage

    GeminiProcessor$extract_usage(response)

#### Arguments

- `response`:

  HTTP response object

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    GeminiProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
