# Consensus Check Optimization (v1.2.4)

## Overview

Starting from version 1.2.4, mllmcelltype implements an optimized consensus checking strategy that reduces LLM API calls while maintaining annotation accuracy.

## How It Works

### Previous Approach (v1.2.3 and earlier)
1. For each cluster, immediately call LLM to analyze all model annotations
2. Fall back to simple consensus only if LLM call fails

### New Optimized Approach (v1.2.4+)
1. **First Stage - Simple Consensus**:
   - Normalize annotations to handle variations (e.g., "T cells" vs "T lymphocytes")
   - Calculate Consensus Proportion (CP) and Shannon Entropy (H)
   - If CP ≥ threshold AND H ≤ threshold → Accept result, skip LLM

2. **Second Stage - LLM Double-Check** (only if needed):
   - Called only when simple consensus doesn't meet thresholds
   - Uses LLM for semantic understanding of ambiguous cases
   - Falls back to simple consensus if LLM fails

## Benefits

- **Cost Reduction**: Fewer LLM API calls for clusters with clear agreement
- **Speed**: Simple consensus is instantaneous compared to LLM calls
- **Same Accuracy**: Clear consensus cases don't need semantic analysis
- **Smart Resource Usage**: LLM only used for genuinely ambiguous cases

## Usage

No code changes required! The optimization works automatically:

```python
from mllmcelltype import check_consensus

# Works exactly the same as before, but faster and cheaper
consensus, cp, entropy, controversial = check_consensus(
    predictions=model_predictions,
    consensus_threshold=0.6,  # Recommended: 0.6
    entropy_threshold=1.2,    # Recommended: 1.2
    api_keys=api_keys
)
```

## Recommended Thresholds

For optimal balance between API savings and accuracy:
- `consensus_threshold`: 0.6 (relaxed from default 0.7)
- `entropy_threshold`: 1.2 (relaxed from default 1.0)

## Monitoring

The package now logs which strategy was used for each cluster:
- `"Cluster X achieved consensus with simple check: CP=Y, H=Z"` - No LLM used
- `"Cluster X needs LLM double-check: CP=Y, H=Z"` - LLM verification needed

## Impact

Based on typical single-cell datasets:
- ~60-70% of clusters have clear consensus among models
- These are now processed without LLM calls
- Results in proportional cost savings
- Processing time reduced significantly