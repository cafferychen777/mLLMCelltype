# mLLMCelltype

[![PyPI version](https://img.shields.io/badge/pypi-v2.0.8-blue.svg)](https://pypi.org/project/mllmcelltype/)
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

**Requirements:** Python >= 3.9, internet connection for LLM API access.

## Quick Start

```python
import pandas as pd
from mllmcelltype import annotate_clusters

marker_genes_df = pd.read_csv('marker_genes.csv')

import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',
    provider='openai',
    model='gpt-5.5',
    tissue='brain'
)

for cluster, annotation in annotations.items():
    print(f"Cluster {cluster}: {annotation}")
```

## Supported Providers and Models

| Provider | Models | API Key Variable |
|----------|--------|-----------------|
| OpenAI | GPT-5.5, GPT-5.4, GPT-5.4-mini | `OPENAI_API_KEY` |
| Anthropic | Claude Opus 4.7, Claude Sonnet 4.6, Claude Haiku 4.5 | `ANTHROPIC_API_KEY` |
| Google | Gemini 3.1 Pro Preview, Gemini 3 Flash Preview, Gemini 3.1 Flash-Lite | `GEMINI_API_KEY` (also supports `GOOGLE_API_KEY`) |
| Alibaba | Qwen3.6 Plus, Qwen3.6 Flash, Qwen3.6 Max Preview | `QWEN_API_KEY` |
| DeepSeek | DeepSeek V4 Flash, DeepSeek V4 Pro | `DEEPSEEK_API_KEY` |
| StepFun | Step 3.5 Flash, Step 3 | `STEPFUN_API_KEY` |
| Zhipu/Z.AI | GLM-5.1, GLM-5, GLM-5-Turbo | `ZHIPU_API_KEY` |
| MiniMax | MiniMax-M2.7, MiniMax-M2.7-highspeed | `MINIMAX_API_KEY` |
| X.AI | Grok-4.3 | `GROK_API_KEY` |
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
from mllmcelltype import format_discussion_report, interactive_consensus_annotation

result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    models=['gpt-5.5', 'claude-sonnet-4-6', 'gemini-3.1-pro-preview', 'qwen3.6-plus'],
    consensus_threshold=0.7,
    max_discussion_rounds=3,
    verbose=True
)

print(result["consensus"])
print(format_discussion_report(result))
```

### Consensus Model Selection

The `consensus_model` parameter specifies which LLM evaluates semantic similarity, calculates consensus metrics, and moderates discussions. Recommended models for consensus checking:

- **Anthropic**: `claude-sonnet-4-6`, `claude-opus-4-7`
- **OpenAI**: `o1`, `gpt-5.5`, `gpt-4.1`
- **Google**: `gemini-3.1-pro-preview`, `gemini-3-flash-preview`
- **Other**: `deepseek-v4-pro`, `qwen3.6-plus`, `grok-4.3`

```python
result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-5.5", "claude-sonnet-4-6", "gemini-3.1-pro-preview"],
    consensus_model="claude-sonnet-4-6",
    consensus_threshold=0.7,
    entropy_threshold=1.0
)
```

If not specified, the consensus checker is selected from the providers available in `api_keys`. Pass `consensus_model` explicitly for reproducible consensus checks.

## Targeted Analysis

### Analyze Specific Clusters

```python
result = interactive_consensus_annotation(
    marker_genes=all_marker_genes,
    species="human",
    models=["gpt-5.5", "claude-sonnet-4-6", "gemini-3.1-pro-preview"],
    clusters_to_analyze=["cluster_0", "cluster_1", "cluster_2"],
    tissue="peripheral blood"
)
```

### Force Fresh Analysis (Bypass Cache)

```python
result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    models=["gpt-5.5", "claude-sonnet-4-6"],
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
    provider='openrouter',
    model='openai/gpt-5.5'
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
    models=['gpt-5.5', 'claude-sonnet-4-6', 'qwen3.6-plus'],
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
    model='gpt-5.5'
)
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)

# Or multi-model consensus
consensus_results = mct.interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    models=['gpt-5.5', 'claude-sonnet-4-6', 'gemini-3.1-pro-preview'],
    consensus_threshold=0.7
)
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus"])
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])
```

See the [examples directory](https://github.com/cafferychen777/mLLMCelltype/tree/main/python/examples) for complete workflow examples.

## Contributing

We welcome contributions. Please submit issues or pull requests on our [GitHub repository](https://github.com/cafferychen777/mLLMCelltype).

### Running tests

Default tests are offline and do not call external APIs:

```bash
python -m pytest
```

Real API integration tests are opt-in:

```bash
python -m pytest --run-integration
```

## License

MIT License

## Citation

If you use mLLMCelltype in your research, please cite:

```bibtex
@article{yang2026llmconsensus,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {Large language model consensus substantially improves the cell type annotation accuracy for scRNA-seq data},
  journal = {Communications Biology},
  year = {2026},
  doi = {10.1038/s42003-026-10420-8},
  url = {https://doi.org/10.1038/s42003-026-10420-8},
  publisher = {Springer Nature}
}
```
