# Grok API Processor

Grok API Processor

Grok API Processor

## Details

Concrete implementation of BaseAPIProcessor for Grok models. Handles
Grok-specific API calls, authentication, and response parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `GrokProcessor`

## Methods

### Public methods

- [`GrokProcessor$new()`](#method-GrokProcessor-new)

- [`GrokProcessor$get_default_api_url()`](#method-GrokProcessor-get_default_api_url)

- [`GrokProcessor$make_api_call()`](#method-GrokProcessor-make_api_call)

- [`GrokProcessor$extract_response_content()`](#method-GrokProcessor-extract_response_content)

- [`GrokProcessor$clone()`](#method-GrokProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize Grok processor

#### Usage

    GrokProcessor$new(base_url = NULL)

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

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Grok API response

#### Usage

    GrokProcessor$extract_response_content(response, model)

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    GrokProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
