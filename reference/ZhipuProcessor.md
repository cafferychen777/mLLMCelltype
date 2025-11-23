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

- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize Zhipu processor

#### Usage

    ZhipuProcessor$new(base_url = NULL)

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

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from Zhipu API response

#### Usage

    ZhipuProcessor$extract_response_content(response, model)

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    ZhipuProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
