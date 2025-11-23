# Print summary of consensus results

This function prints a detailed summary of the consensus building
process, including initial predictions from all models, uncertainty
metrics, and final consensus for each controversial cluster.

## Usage

``` r
print_consensus_summary(results)
```

## Details

- initial_results: A list containing individual_predictions,
  consensus_results, and controversial_clusters

- final_annotations: A list of final cell type annotations for each
  cluster

- controversial_clusters: A character vector of cluster IDs that were
  controversial

- discussion_logs: A list of discussion logs for each controversial
  cluster
