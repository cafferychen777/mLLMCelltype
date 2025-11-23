# Check if consensus is reached among models

Check if consensus is reached among models

## Usage

``` r
check_consensus(
  round_responses,
  api_keys = NULL,
  controversy_threshold = 2/3,
  entropy_threshold = 1,
  consensus_check_model = NULL
)
```

## Note

This function uses create_consensus_check_prompt from prompt_templates.R
