# Clear mLLMCelltype cache

Clear the mLLMCelltype cache

## Usage

``` r
mllmcelltype_clear_cache(cache_dir = NULL)
```

## Arguments

- cache_dir:

  Cache directory specification. NULL uses system default, "local" uses
  current dir, "temp" uses temp dir, or custom path

## Value

Invisible NULL

## Examples

``` r
if (FALSE) { # \dontrun{
mllmcelltype_clear_cache()
mllmcelltype_clear_cache("local")
} # }
```
