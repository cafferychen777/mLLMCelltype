# Zhipu API Processor

Zhipu API Processor

Zhipu API Processor

## Details

Concrete implementation of BaseAPIProcessor for Zhipu models. Handles
Zhipu-specific API calls, authentication, and response parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `ZhipuProcessor`

## Methods

### Public methods

- [`ZhipuProcessor$new()`](#method-ZhipuProcessor-new)

- [`ZhipuProcessor$get_default_api_url()`](#method-ZhipuProcessor-get_default_api_url)

- [`ZhipuProcessor$make_api_call()`](#method-ZhipuProcessor-make_api_call)

- [`ZhipuProcessor$extract_response_content()`](#method-ZhipuProcessor-extract_response_content)

- [`ZhipuProcessor$clone()`](#method-ZhipuProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$extract_usage()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-extract_usage)
- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize Zhipu processor

#### Usage

    ZhipuProcessor$new(base_url = NULL)

#### Arguments

- `base_url`:

  Optional custom API endpoint

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default Zhipu API URL

#### Usage

    ZhipuProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to Zhipu

#### Usage

    ZhipuProcessor$make_api_call(chunk_content, model, api_key)

#### Arguments

- `chunk_content`:

  Prompt text to send

- `model`:

  Model identifier

- `api_key`:

  Zhipu API key

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Zhipu API response

#### Usage

    ZhipuProcessor$extract_response_content(response, model)

#### Arguments

- `response`:

  HTTP response object

- `model`:

  Model identifier

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    ZhipuProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
