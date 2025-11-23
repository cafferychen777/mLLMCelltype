# Register a custom model for a provider

Register a custom model for a provider

## Usage

``` r
register_custom_model(model_name, provider_name, model_config = list())
```

## Arguments

- model_name:

  Unique name for the custom model

- provider_name:

  Name of the provider this model belongs to

- model_config:

  List of configuration parameters for the model (e.g., temperature,
  max_tokens)

## Value

Invisible TRUE on success

## Examples

``` r
if (FALSE) { # \dontrun{
register_custom_model(
  model_name = "my_model",
  provider_name = "my_provider",
  model_config = list(
    temperature = 0.7,
    max_tokens = 2000
  )
)
} # }
```
