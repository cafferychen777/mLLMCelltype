# Utility functions for API key management

This file contains utility functions for managing API keys and related
operations. Get API key for a specific model

## Usage

``` r
get_api_key(model, api_keys)
```

## Arguments

- model:

  Model name to get API key for

- api_keys:

  Named list of API keys with provider or model names as keys

## Value

API key string for the specified model

## Details

This function retrieves the appropriate API key for a given model by
first checking the provider name and then the model name in the provided
API keys list.
