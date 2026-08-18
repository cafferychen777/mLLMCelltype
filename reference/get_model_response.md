# Get response from a specific model

Get response from a specific model

## Usage

``` r
get_model_response(prompt, model, api_key, base_urls = NULL, normalize = TRUE)
```

## Arguments

- prompt:

  Non-empty prompt string

- model:

  Non-empty model name

- api_key:

  Non-empty API key

- base_urls:

  Optional shared or provider-specific base URL configuration

- normalize:

  Logical; if `TRUE` (default) the response is normalized into non-empty
  lines, otherwise the raw response string is returned.

## Value

Provider response as a character vector
