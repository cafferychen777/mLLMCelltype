# Minimax API Processor

Minimax API Processor

Minimax API Processor

## Details

Concrete implementation of BaseAPIProcessor for Minimax models. Handles
Minimax-specific API calls, authentication, and response parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `MinimaxProcessor`

## Methods

### Public methods

- [`MinimaxProcessor$new()`](#method-MinimaxProcessor-new)

- [`MinimaxProcessor$get_default_api_url()`](#method-MinimaxProcessor-get_default_api_url)

- [`MinimaxProcessor$make_api_call()`](#method-MinimaxProcessor-make_api_call)

- [`MinimaxProcessor$extract_response_content()`](#method-MinimaxProcessor-extract_response_content)

- [`MinimaxProcessor$clone()`](#method-MinimaxProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$extract_usage()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-extract_usage)
- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize Minimax processor

#### Usage

    MinimaxProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default MiniMax OpenAI-compatible chat completions API URL

#### Usage

    MinimaxProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to Minimax

#### Usage

    MinimaxProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier

- `api_key`:

  MiniMax API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Minimax API response

#### Usage

    MinimaxProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    MinimaxProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
