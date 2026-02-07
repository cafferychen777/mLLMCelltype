# Parse text-format model predictions into a named list

Handles multiple output formats from LLMs:

- "cluster_id: cell_type" format

- "1. cell_type" numeric index format

- Positional fallback (line index maps to cluster index)

## Usage

``` r
parse_text_predictions(model_preds, all_clusters = NULL)
```

## Arguments

- model_preds:

  Character vector of prediction lines from a model

- all_clusters:

  Optional character vector of cluster IDs for positional fallback

## Value

Named list mapping cluster_id -\> cell_type
