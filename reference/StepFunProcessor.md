# StepFun API Processor

StepFun API Processor

StepFun API Processor

## Details

Concrete implementation of BaseAPIProcessor for StepFun models. Handles
StepFun-specific API calls, authentication, and response parsing.

## Super class

[`mLLMCelltype::BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
-\> `StepFunProcessor`

## Methods

### Public methods

- [`StepFunProcessor$new()`](#method-StepFunProcessor-new)

- [`StepFunProcessor$get_default_api_url()`](#method-StepFunProcessor-get_default_api_url)

- [`StepFunProcessor$make_api_call()`](#method-StepFunProcessor-make_api_call)

- [`StepFunProcessor$extract_response_content()`](#method-StepFunProcessor-extract_response_content)

- [`StepFunProcessor$clone()`](#method-StepFunProcessor-clone)

Inherited methods

- [`mLLMCelltype::BaseAPIProcessor$get_api_url()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-get_api_url)
- [`mLLMCelltype::BaseAPIProcessor$process_request()`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.html#method-process_request)

------------------------------------------------------------------------

### Method `new()`

Initialize StepFun processor

#### Usage

    StepFunProcessor$new(base_url = NULL)

------------------------------------------------------------------------

### Method `get_default_api_url()`

Get default StepFun API URL

#### Usage

    StepFunProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Make API call to StepFun

#### Usage

    StepFunProcessor$make_api_call(chunk_content, model, api_key)

------------------------------------------------------------------------

### Method `extract_response_content()`

Extract response content from StepFun API response

#### Usage

    StepFunProcessor$extract_response_content(response, model)

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    StepFunProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
