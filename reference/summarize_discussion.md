# Summarize discussion and determine final cell type

NOTE: This function is currently not in use. The consensus_annotation.R
file now directly extracts the majority_prediction from the last round
of discussion. This function is kept for potential future use or
reference.

## Usage

``` r
summarize_discussion(discussion_log, cluster_id, model, api_key)
```

## Arguments

- discussion_log:

  Discussion log for a cluster

- cluster_id:

  Cluster identifier

- model:

  Model to use for summary

- api_key:

  API key for the model

## Value

Final cell type determination
