# Get mLLMCelltype cache location

Display the cache directory location

## Usage

``` r
mllmcelltype_cache_dir(cache_dir = NULL)
```

## Arguments

- cache_dir:

  Cache directory specification. NULL uses system default, "local" uses
  current dir, "temp" uses temp dir, or custom path

## Value

Invisible cache directory path

## Examples

``` r
if (FALSE) { # \dontrun{
mllmcelltype_cache_dir()
mllmcelltype_cache_dir("local")
} # }
```
