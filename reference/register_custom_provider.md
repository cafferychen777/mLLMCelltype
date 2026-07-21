# Register a custom LLM provider

Register a custom LLM provider

## Usage

``` r
register_custom_provider(provider_name, process_fn, description = NULL)
```

## Arguments

- provider_name:

  Unique name for the custom provider

- process_fn:

  Function that processes LLM requests. Must accept parameters: prompt,
  model, api_key; may optionally accept model_config and base_url

- description:

  Optional description of the provider

## Value

Invisible NULL

## Examples

``` r
if (FALSE) { # \dontrun{
register_custom_provider(
  provider_name = "my_provider",
  process_fn = function(prompt, model, api_key) {
    # Custom implementation
    response <- httr::POST(
      url = "your_api_endpoint",
      body = list(prompt = prompt),
      encode = "json"
    )
    return(httr::content(response)$choices[[1]]$text)
  }
)
} # }
```
