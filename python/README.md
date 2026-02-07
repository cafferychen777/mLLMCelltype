# mLLMCelltype

[![PyPI version](https://img.shields.io/badge/pypi-v1.1.0-blue.svg)](https://pypi.org/project/mllmcelltype/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

mLLMCelltype is a Python package for cell type annotation in single-cell RNA sequencing data using an iterative multi-LLM consensus approach. It combines predictions from multiple large language models and provides uncertainty quantification. The package is compatible with the scverse ecosystem, including AnnData objects and Scanpy workflows.

## Installation

```bash
pip install mllmcelltype
```

For development:

```bash
git clone https://github.com/cafferychen777/mLLMCelltype.git
cd mLLMCelltype/python
pip install -e .
```

**Requirements:** Python >= 3.8, internet connection for LLM API access.

## Quick Start

```python
import pandas as pd
from mllmcelltype import annotate_clusters, setup_logging

setup_logging()

marker_genes_df = pd.read_csv('marker_genes.csv')

import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',
    provider='openai',
    model='gpt-5.2',
    tissue='brain'
)

for cluster, annotation in annotations.items():
    print(f"Cluster {cluster}: {annotation}")
```

## Supported Providers and Models

| Provider | Models | API Key Variable |
|----------|--------|-----------------|
| OpenAI | GPT-5.2, GPT-5, O3-Pro, etc. | `OPENAI_API_KEY` |
| Anthropic | Claude 4.6 Opus, Claude 4.5 Sonnet/Haiku, etc. | `ANTHROPIC_API_KEY` |
| Google | Gemini 3 Pro, Gemini 3 Flash, etc. | `GOOGLE_API_KEY` |
| Alibaba | Qwen3-Max, Qwen-Plus, etc. | `QWEN_API_KEY` |
| DeepSeek | DeepSeek-Chat, DeepSeek-Reasoner | `DEEPSEEK_API_KEY` |
| StepFun | Step-3, Step-2-16k, Step-2-Mini | `STEPFUN_API_KEY` |
| Zhipu AI | GLM-4.7, GLM-4-Plus | `ZHIPU_API_KEY` |
| MiniMax | MiniMax-M2.1, MiniMax-M2 | `MINIMAX_API_KEY` |
| X.AI | Grok-4 | `GROK_API_KEY` |
| OpenRouter | Access to multiple models via single API | `OPENROUTER_API_KEY` |

API keys can be set via environment variables, passed directly as parameters, or loaded from a `.env` file.

## Annotation Features

- **Iterative Consensus**: Multiple rounds of comparison between LLM outputs to resolve disagreements
- **Uncertainty Quantification**: Consensus Proportion (CP) and Shannon Entropy (H) metrics
- **Cross-model Comparison**: Helps identify inconsistent predictions across models
- **Hierarchical Annotation**: Optional multi-resolution analysis with parent-child consistency
- **Caching**: Avoids redundant API calls to reduce costs
- **Custom Base URLs**: Configure custom API endpoints for proxy servers or enterprise gateways

## Multi-LLM Consensus Annotation

```python
from mllmcelltype import interactive_consensus_annotation, print_consensus_summary

result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    models=['gpt-5.2', 'claude-sonnet-4-5-20250929', 'gemini-3-pro', 'qwen3-max'],
    consensus_threshold=0.7,
    max_discussion_rounds=3,
    verbose=True
)

print_consensus_summary(result)
```

### Consensus Model Selection

The `consensus_model` parameter specifies which LLM evaluates semantic similarity, calculates consensus metrics, and moderates discussions. Recommended models for consensus checking:

- **Anthropic**: `claude-sonnet-4-5-20250929`, `claude-opus-4-1-20250805`
- **OpenAI**: `o1`, `gpt-5.2`, `gpt-4.1`
- **Google**: `gemini-3-pro`, `gemini-3-flash`
- **Other**: `deepseek-r1`, `qwen3-max`, `grok-4`

```python
result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-pro"],
    consensus_model="claude-sonnet-4-5-20250929",
    consensus_threshold=0.7,
    entropy_threshold=1.0
)
```

If not specified, defaults to `qwen3-max` with `claude-sonnet-4-5-20250929` as fallback.

## Targeted Analysis

### Analyze Specific Clusters

```python
result = interactive_consensus_annotation(
    marker_genes=all_marker_genes,
    species="human",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-pro"],
    clusters_to_analyze=["cluster_0", "cluster_1", "cluster_2"],
    tissue="peripheral blood"
)
```

### Force Fresh Analysis (Bypass Cache)

```python
result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    models=["gpt-5.2", "claude-sonnet-4-5-20250929"],
    tissue="peripheral blood",
    additional_context="Patient with autoimmune disease",
    force_rerun=True
)
```

## OpenRouter Integration

OpenRouter provides a unified API for accessing models from multiple providers.

```python
from mllmcelltype import annotate_clusters

annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider_config={"provider": "openrouter", "model": "openai/gpt-5.2"}
)
```

Free models are available with the `:free` suffix (e.g., `meta-llama/llama-4-maverick:free`).

## Custom Base URL Configuration

```python
base_urls = {
    'openai': 'https://openai-proxy.com/v1/chat/completions',
    'anthropic': 'https://anthropic-proxy.com/v1/messages',
    'qwen': 'https://qwen-proxy.com/compatible-mode/v1/chat/completions'
}

result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    models=['gpt-5.2', 'claude-3-opus', 'qwen-max'],
    api_keys=your_api_keys,
    base_urls=base_urls
)
```

For Qwen models, endpoint selection between international and domestic endpoints is handled automatically.

## Scanpy/AnnData Integration

```python
import scanpy as sc
import mllmcelltype as mct

# After standard preprocessing and clustering...
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
marker_genes = {}
for cluster in adata.obs['leiden'].unique():
    genes = sc.get.rank_genes_groups_df(adata, group=cluster)['names'].tolist()[:20]
    marker_genes[cluster] = genes

# Single model annotation
annotations = mct.annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    provider='openai',
    model='gpt-5.2'
)
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)

# Or multi-model consensus
consensus_results = mct.interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    models=['gpt-5.2', 'claude-sonnet-4-5-20250929', 'gemini-3-pro'],
    consensus_threshold=0.7
)
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus"])
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])
```

See the [examples directory](https://github.com/cafferychen777/mLLMCelltype/tree/main/python/examples) for complete workflow examples.

## Contributing

We welcome contributions. Please submit issues or pull requests on our [GitHub repository](https://github.com/cafferychen777/mLLMCelltype).

## License

MIT License

## Citation

If you use mLLMCelltype in your research, please cite:

```bibtex
@article{Yang2025.04.10.647852,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data},
  elocation-id = {2025.04.10.647852},
  year = {2025},
  doi = {10.1101/2025.04.10.647852},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2025/04/17/2025.04.10.647852},
  journal = {bioRxiv}
}
```
