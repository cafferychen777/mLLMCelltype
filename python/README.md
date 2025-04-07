# LLMCelltype

A Python module for cell type annotation using various Large Language Models (LLMs).

## Overview

LLMCelltype is a Python implementation of the R package with the same name, designed to annotate cell types in single-cell RNA sequencing data using various Large Language Models (LLMs). This package provides a unified interface to multiple LLM providers, making it easy to leverage different models for cell type annotation.

## Features

- Support for multiple LLM providers:
  - OpenAI (GPT-4o, GPT-4, etc.)
  - Anthropic (Claude 3 Opus, Claude 3 Sonnet, etc.)
  - DeepSeek
  - Google Gemini
  - Alibaba Qwen
  - StepFun
  - Zhipu AI (ChatGLM)
  - MiniMax
- Simple and consistent API for all providers
- Multi-model consensus annotation for improved accuracy
- Automatic resolution of controversial cluster annotations
- Model prediction comparison and analysis tools
- Visualization of model agreement patterns
- Caching of results to avoid redundant API calls
- Comprehensive logging
- Customizable prompts
- Structured JSON responses with confidence scores

## Installation

```bash
pip install llmcelltype
```

For development installation:

```bash
git clone https://github.com/cafferychen777/LLMCelltype.git
cd LLMCelltype/code/python
pip install -e .
```

## Quick Start

```python
import pandas as pd
from llmcelltype import annotate_clusters, setup_logging

# Setup logging
setup_logging()

# Load marker genes
marker_genes_df = pd.read_csv('marker_genes.csv')

# Annotate clusters
annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',
    provider='openai',
    model='gpt-4o',
    tissue='brain'
)

# Print annotations
for cluster, annotation in annotations.items():
    print(f"Cluster {cluster}: {annotation}")
```

## API Keys

LLMCelltype requires API keys for the LLM providers you want to use. You can set these as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
# ... and so on for other providers
```

Alternatively, you can pass the API key directly to the `annotate_clusters` function:

```python
annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',
    provider='openai',
    api_key='your-openai-api-key'
)
```

## Advanced Usage

### Batch Annotation

```python
from llmcelltype import batch_annotate_clusters

# Prepare multiple sets of marker genes
marker_genes_list = [marker_genes_df1, marker_genes_df2, marker_genes_df3]

# Batch annotate
batch_annotations = batch_annotate_clusters(
    marker_genes_list=marker_genes_list,
    species='mouse',
    provider='anthropic',
    model='claude-3-opus-20240229'
)

# Process results
for i, annotations in enumerate(batch_annotations):
    print(f"Set {i+1}:")
    for cluster, annotation in annotations.items():
        print(f"  Cluster {cluster}: {annotation}")
```

### Consensus Annotation

```python
from llmcelltype import interactive_consensus_annotation, print_consensus_summary

# Define marker genes
marker_genes = {
    "1": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],
    "2": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],
    "3": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]
}

# Run consensus annotation with multiple models
result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    models=['gpt-4o', 'claude-3-opus', 'gemini-1.5-pro'],
    consensus_threshold=0.6,
    verbose=True
)

# Print consensus summary
print_consensus_summary(result)
```

### Model Comparison

```python
from llmcelltype import compare_model_predictions, create_comparison_table

# Compare results from different models
model_predictions = {
    "OpenAI (GPT-4o)": results_openai,
    "Anthropic (Claude)": results_claude,
    "Google (Gemini)": results_gemini
}

# Compare model predictions
agreement_df, metrics = compare_model_predictions(
    model_predictions=model_predictions,
    display_plot=True  # Set to True to display a heatmap
)

# Print agreement metrics
print(f"Average agreement: {metrics['agreement_avg']:.2f}")

# Create and display a comparison table
comparison_table = create_comparison_table(model_predictions)
print(comparison_table)
```

### Custom Prompts

```python
from llmcelltype import annotate_clusters

# Define custom prompt template
custom_template = """You are an expert in single-cell RNA sequencing analysis.
Please annotate the following cell clusters based on their marker genes.

{context}

Marker genes:
{clusters}

Provide only the cell type name for each cluster, one per line.
"""

# Annotate with custom prompt
annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',
    provider='openai',
    prompt_template=custom_template
)
```

### JSON Response Format

LLMCelltype now supports structured JSON responses, which can provide more detailed information about each annotation:

```python
from llmcelltype import annotate_clusters

# Define JSON prompt template
json_template = """You are a cell type annotation expert. Below are marker genes for different cell clusters in {context}.

{clusters}

For each numbered cluster, provide the cell type annotation in JSON format. 
Use the following structure:
```json
{
  "annotations": [
    {
      "cluster": "1",
      "cell_type": "cell type name",
      "confidence": "high/medium/low",
      "key_markers": ["marker1", "marker2", "marker3"]
    },
    ...
  ]
}
```
"""

# Annotate with JSON prompt
json_annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',
    provider='openai',
    prompt_template=json_template
)

# The parser will automatically extract the cell type annotations from the JSON response
# But the raw JSON response is also available in the cache for advanced processing
```

Using JSON responses provides several advantages:
- Structured data that can be easily processed
- Additional metadata like confidence levels and key markers
- More consistent parsing across different LLM providers

## License

MIT License

## Citation

If you use LLMCelltype in your research, please cite:

```
@software{llmcelltype,
  author = {LLMCelltype Team},
  title = {LLMCelltype: A Python module for cell type annotation using various LLMs},
  url = {https://github.com/cafferychen777/LLMCelltype},
  year = {2025}
}
```
