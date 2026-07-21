# Parse text-format model predictions into a named list

Reads a response as explicit cluster labels (keyed by resolved cluster
ID, so out-of-order labels land correctly) when at least one label
resolves, unless an unresolved cluster-reference key coexists with
incomplete coverage – that signals list ordinals misaligned against the
requested clusters and falls through to positional mapping. Positional
mapping preserves the line-\>cluster slot correspondence so a mid-list
"Unknown" does not shift later clusters, and a stray "Summary:"/"Note:"
line does not hijack the parse.

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

Named list mapping cluster IDs to cell type annotations
