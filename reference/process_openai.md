# Process request using OpenAI models

Main function that creates an OpenAI processor and handles the request.
This maintains backward compatibility with the existing API.

This function uses the new BaseAPIProcessor architecture for improved
maintainability and consistent logging across all API providers.

## Usage

``` r
process_openai(prompt, model, api_key, base_url = NULL)

process_openai(prompt, model, api_key, base_url = NULL)
```
