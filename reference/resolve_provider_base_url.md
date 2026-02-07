# URL Utilities for Base URL Resolution

This file contains utility functions for resolving custom base URLs for
different API providers. Resolve provider-specific base URL

## Usage

``` r
resolve_provider_base_url(provider, base_urls)
```

## Arguments

- provider:

  Provider name (e.g., "openai", "anthropic")

- base_urls:

  User-provided base URLs: NULL, a single string, or a named list

## Value

Resolved and normalized base URL, or NULL if not specified

## Details

This is the single entry point for all base URL resolution. It resolves
the appropriate URL and normalizes it (strips trailing slashes).
