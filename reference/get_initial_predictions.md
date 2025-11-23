# Get initial predictions from all models

This function retrieves initial cell type predictions from all specified
models. It is an internal helper function used by the
interactive_consensus_annotation function.

## Usage

``` r
get_initial_predictions(
  input,
  tissue_name,
  models,
  api_keys,
  top_gene_count,
  base_urls = NULL
)
```
