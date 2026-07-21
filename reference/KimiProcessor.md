# Kimi API Processor

Kimi API Processor

Kimi API Processor

## Details

Concrete implementation of BaseAPIProcessor for Kimi models. By default
it targets the Moonshot AI Open Platform over the OpenAI-compatible Chat
Completions protocol, with k2 thinking mode disabled for deterministic
output. A custom `base_url` may instead point at the Kimi Code platform
(api.kimi.com/coding), which speaks both protocols; the protocol is
inferred from the effective endpoint URL. URLs ending in '/messages' use
the Anthropic Messages protocol; the Kimi Code base
'https://api.kimi.com/coding' and URLs ending in '/chat/completions' use
OpenAI-compatible Chat Completions.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `KimiProcessor`

## Methods

### Public methods

- [`KimiProcessor$new()`](#method-KimiProcessor-new)

- [`KimiProcessor$get_default_api_url()`](#method-KimiProcessor-get_default_api_url)

- [`KimiProcessor$make_api_call()`](#method-KimiProcessor-make_api_call)

- [`KimiProcessor$extract_response_content()`](#method-KimiProcessor-extract_response_content)

- [`KimiProcessor$extract_usage()`](#method-KimiProcessor-extract_usage)

- [`KimiProcessor$clone()`](#method-KimiProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize Kimi processor

#### Usage

    KimiProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default Kimi API URL

#### Usage

    KimiProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to Kimi

#### Usage

    KimiProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier (e.g., 'kimi-k2.6', 'moonshot-v1-8k')

- `api_key`:

  Moonshot API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Kimi API response

#### Usage

    KimiProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `extract_usage()`

Extract normalized token usage from a Kimi API response

#### Usage

    KimiProcessor$extract_usage(response)

#### Arguments

- `response`:

  HTTP response object

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    KimiProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
