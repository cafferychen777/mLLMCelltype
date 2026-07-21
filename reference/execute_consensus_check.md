# Execute consensus check across ordered model candidates

Execute consensus check across ordered model candidates

## Usage

``` r
execute_consensus_check(
  formatted_responses,
  api_keys,
  models_to_try,
  base_urls = NULL
)
```

## Arguments

- formatted_responses:

  Prompt containing model responses

- api_keys:

  Named API key list

- models_to_try:

  Ordered model candidates

- base_urls:

  Optional provider base URLs

## Value

A list containing success status and response text
