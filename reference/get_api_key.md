# Get an API key for a model

Retrieves a configured API key by checking the model's provider name
first, followed by the exact model name.

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

A trimmed API key string, or `NULL` when no valid key is configured.
