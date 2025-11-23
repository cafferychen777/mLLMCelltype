# Facilitate discussion for a controversial cluster

Facilitate discussion for a controversial cluster

## Usage

``` r
facilitate_cluster_discussion(
  cluster_id,
  input,
  tissue_name,
  models,
  api_keys,
  initial_predictions,
  top_gene_count,
  max_rounds = 3,
  controversy_threshold = 0.7,
  entropy_threshold = 1,
  consensus_check_model = NULL
)
```

## Note

This function uses create_initial_discussion_prompt and
create_discussion_prompt from prompt_templates.R
