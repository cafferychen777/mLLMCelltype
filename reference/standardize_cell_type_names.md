# Standardize cell type names using a language model

This function takes predictions from multiple models and standardizes
the cell type nomenclature to ensure consistent naming across different
models' outputs.

## Usage

``` r
standardize_cell_type_names(
  predictions,
  models,
  api_keys,
  standardization_model = "claude-sonnet-4-20250514",
  base_urls = NULL
)
```

## Details

1.  With provider names as keys:
    `list("openai" = "sk-...", "anthropic" = "sk-ant-...", "openrouter" = "sk-or-...")`

2.  With model names as keys:
    `list("gpt-5" = "sk-...", "claude-sonnet-4-5-20250929" = "sk-ant-...")`
