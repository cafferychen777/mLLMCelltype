# Iterative Discussion Feature Guide

This document introduces the iterative discussion feature newly added to the LLMCelltype framework, which allows multiple rounds of discussion to resolve controversial cell cluster annotations.

## Feature Overview

The iterative discussion feature allows the LLMCelltype framework to conduct multiple rounds of in-depth discussion when dealing with controversial cell clusters, rather than relying on a single round of discussion. This approach is consistent with the functionality in the R version and can better resolve complex cell type annotation problems.

Main features:
- Support for multiple rounds of discussion (configurable maximum discussion rounds)
- Check for consensus after each round of discussion
- Quantify annotation certainty using Consensus Proportion (CP) and Shannon Entropy (H)
- Save complete discussion logs for subsequent analysis

## Usage

### Basic Usage

When calling the `interactive_consensus_annotation` function, add the `max_discussion_rounds` parameter:

```python
result = lct.interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    tissue='lung',
    models=models,
    api_keys=api_keys,
    consensus_threshold=0.6,
    max_discussion_rounds=3,  # Set maximum discussion rounds to 3
    use_cache=True,
    verbose=True
)
```

### View Discussion Logs

Discussion logs are stored in the `discussion_logs` key of the result dictionary:

```python
# Print discussion logs for a specific cell cluster
if "discussion_logs" in result and "4" in result["discussion_logs"]:
    print("\n=== Discussion logs for cell cluster #4 ===\n")
    print(result["discussion_logs"]["4"])
```

### Example Code

We provide two example scripts:
1. `consensus_example.py` - Basic consensus annotation example, including iterative discussion functionality
2. `iterative_discussion_example.py` - Detailed example specifically demonstrating the iterative discussion feature

Run example:
```bash
python examples/iterative_discussion_example.py
```

## Consensus Metrics Explanation

The iterative discussion feature uses two main metrics to quantify the certainty of annotations:

1. **Consensus Proportion (CP)**
   - Range: 0-1
   - Meaning: Proportion of evidence supporting the main cell type
   - Higher values indicate more consistent evidence support

2. **Shannon Entropy (H)**
   - Range: 0+ (theoretically unlimited, but actual values are usually small)
   - Meaning: Uncertainty of annotation
   - Lower values indicate less uncertainty

These metrics are dynamically updated during the discussion process, helping to track the formation of consensus.

## Parameter Adjustment Suggestions

- `max_discussion_rounds`: Usually set to 2-5 rounds. Too many rounds may lead to redundant discussions, while too few rounds may not adequately resolve disputes.
- `consensus_threshold`: Recommended to be set between 0.5-0.7. Higher thresholds will cause more cell clusters to be marked as controversial, thus entering the discussion process.

## Notes

1. Iterative discussions will increase the number of API calls, which may increase costs.
2. Using `use_cache=True` can avoid repeated API calls, especially during development and testing phases.
3. Discussion logs can be lengthy; it is recommended to selectively view logs when processing a large number of cell clusters.
