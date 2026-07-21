# Anthropic API Processor

Anthropic API Processor

Anthropic API Processor

## Details

Concrete implementation of BaseAPIProcessor for Anthropic models.
Handles Anthropic-specific API calls, authentication, and response
parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `AnthropicProcessor`

## Methods

### Public methods

- [`AnthropicProcessor$new()`](#method-AnthropicProcessor-new)

- [`AnthropicProcessor$get_default_api_url()`](#method-AnthropicProcessor-get_default_api_url)

- [`AnthropicProcessor$make_api_call()`](#method-AnthropicProcessor-make_api_call)

- [`AnthropicProcessor$extract_response_content()`](#method-AnthropicProcessor-extract_response_content)

- [`AnthropicProcessor$extract_usage()`](#method-AnthropicProcessor-extract_usage)

- [`AnthropicProcessor$clone()`](#method-AnthropicProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize Anthropic processor

#### Usage

    AnthropicProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default Anthropic API URL

#### Usage

    AnthropicProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to Anthropic

#### Usage

    AnthropicProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier

- `api_key`:

  Anthropic API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Anthropic API response

#### Usage

    AnthropicProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `extract_usage()`

Extract normalized Anthropic token usage

#### Usage

    AnthropicProcessor$extract_usage(response)

#### Arguments

- `response`:

  HTTP response object

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    AnthropicProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
