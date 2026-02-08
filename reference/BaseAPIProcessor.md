# Base API Processor Class

Base API Processor Class

Base API Processor Class

## Details

Abstract base class for API processors that provides common
functionality including unified logging, error handling, input
processing, and response validation. This eliminates code duplication
across all provider-specific processors.

## Public fields

- `provider_name`:

  Name of the API provider

- `logger`:

  Unified logger instance

- `base_url`:

  Custom base URL for API endpoints

## Methods

### Public methods

- [`BaseAPIProcessor$new()`](#method-BaseAPIProcessor-new)

- [`BaseAPIProcessor$process_request()`](#method-BaseAPIProcessor-process_request)

- [`BaseAPIProcessor$get_api_url()`](#method-BaseAPIProcessor-get_api_url)

- [`BaseAPIProcessor$get_default_api_url()`](#method-BaseAPIProcessor-get_default_api_url)

- [`BaseAPIProcessor$make_api_call()`](#method-BaseAPIProcessor-make_api_call)

- [`BaseAPIProcessor$extract_response_content()`](#method-BaseAPIProcessor-extract_response_content)

- [`BaseAPIProcessor$clone()`](#method-BaseAPIProcessor-clone)

------------------------------------------------------------------------

### Method `new()`

Initialize the base API processor

#### Usage

    BaseAPIProcessor$new(provider_name, base_url = NULL)

------------------------------------------------------------------------

### Method `process_request()`

Main entry point for processing API requests

#### Usage

    BaseAPIProcessor$process_request(prompt, model, api_key)

------------------------------------------------------------------------

### Method `get_api_url()`

Get the API URL to use for requests

#### Usage

    BaseAPIProcessor$get_api_url()

------------------------------------------------------------------------

### Method `get_default_api_url()`

Abstract method to be implemented by subclasses for getting default API
URL

#### Usage

    BaseAPIProcessor$get_default_api_url()

------------------------------------------------------------------------

### Method `make_api_call()`

Abstract method to be implemented by subclasses for making the actual
API call

#### Usage

    BaseAPIProcessor$make_api_call(chunk_content, model, api_key)

------------------------------------------------------------------------

### Method `extract_response_content()`

Abstract method to be implemented by subclasses for extracting content
from response Make API call and extract response content

#### Usage

    BaseAPIProcessor$extract_response_content(response, model)

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    BaseAPIProcessor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
