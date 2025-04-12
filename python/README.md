# mLLMCelltype

[![PyPI version](https://img.shields.io/badge/pypi-v1.0.0-blue.svg)](https://pypi.org/project/mllmcelltype/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

mLLMCelltype is a comprehensive Python framework for automated cell type annotation in single-cell RNA sequencing data through an iterative multi-LLM consensus approach. By leveraging the collective intelligence of multiple large language models, this framework significantly improves annotation accuracy while providing robust uncertainty quantification.

### Scientific Background

Single-cell RNA sequencing has revolutionized our understanding of cellular heterogeneity, but accurate cell type annotation remains challenging. Traditional annotation methods often rely on reference datasets or manual expert curation, which can be time-consuming and subjective. mLLMCelltype addresses these limitations by implementing a novel multi-model deliberative framework that:

1. Harnesses complementary strengths of diverse LLMs to overcome single-model limitations
2. Implements a structured deliberation process for collaborative reasoning
3. Provides quantitative uncertainty metrics to identify ambiguous annotations
4. Maintains high accuracy even with imperfect marker gene inputs

## Key Features

### Multi-LLM Architecture
- **Comprehensive Provider Support**:
  - OpenAI (GPT-4o, O1, etc.)
  - Anthropic (Claude 3.7 Sonnet, Claude 3.5 Haiku, etc.)
  - Google (Gemini 2.5 Pro, Gemini 2.0 Flash, etc.)
  - Alibaba (Qwen-Max, etc.)
  - DeepSeek (DeepSeek-Chat, DeepSeek-Reasoner)
  - StepFun (Step-2-Mini, Step-2-16k, etc.)
  - Zhipu AI (GLM-4-Plus, GLM-3-Turbo)
  - MiniMax (MiniMax-Text-01)

### Advanced Annotation Capabilities
- **Iterative Consensus Framework**: Enables multiple rounds of structured deliberation between LLMs
- **Uncertainty Quantification**: Provides Consensus Proportion (CP) and Shannon Entropy (H) metrics
- **Hallucination Reduction**: Cross-model verification minimizes unsupported predictions
- **Hierarchical Annotation**: Optional support for multi-resolution analysis with parent-child consistency

### Technical Features
- **Unified API**: Consistent interface across all LLM providers
- **Intelligent Caching**: Avoids redundant API calls to reduce costs and improve performance
- **Comprehensive Logging**: Captures full deliberation process for transparency and debugging
- **Structured JSON Responses**: Standardized output format with confidence scores
- **Seamless Integration**: Works directly with Scanpy/AnnData workflows

## Installation

### PyPI Installation (Recommended)

```bash
pip install mllmcelltype
```

### Development Installation

```bash
git clone https://github.com/cafferychen777/mLLMCelltype.git
cd mLLMCelltype/python
pip install -e .
```

### System Requirements

- Python ≥ 3.8
- Dependencies are automatically installed with the package
- Internet connection for API access to LLM providers

## Quick Start

```python
import pandas as pd
from mllmcelltype import annotate_clusters, setup_logging

# Setup logging (optional but recommended)
setup_logging()

# Load marker genes (from Scanpy, Seurat, or other sources)
marker_genes_df = pd.read_csv('marker_genes.csv')

# Configure API keys (alternatively use environment variables)
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Annotate clusters with a single model
annotations = annotate_clusters(
    marker_genes=marker_genes_df,  # DataFrame or dictionary of marker genes
    species='human',               # Organism species
    provider='openai',            # LLM provider
    model='gpt-4o',               # Specific model
    tissue='brain'                # Tissue context (optional but recommended)
)

# Print annotations
for cluster, annotation in annotations.items():
    print(f"Cluster {cluster}: {annotation}")
```

## API Authentication

mLLMCelltype requires API keys for the LLM providers you intend to use. These can be configured in several ways:

### Environment Variables (Recommended)

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export QWEN_API_KEY="your-qwen-api-key"
# Additional providers as needed
```

### Direct Parameter

```python
annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',
    provider='openai',
    api_key='your-openai-api-key'  # Direct API key parameter
)
```

### Configuration File

```python
from mllmcelltype import load_api_key

# Load from .env file or custom config
load_api_key(provider='openai', path='.env')
```

## Advanced Usage

### Batch Annotation

```python
from mllmcelltype import batch_annotate_clusters

# Prepare multiple sets of marker genes (e.g., from different samples)
marker_genes_list = [marker_genes_df1, marker_genes_df2, marker_genes_df3]

# Batch annotate multiple datasets efficiently
batch_annotations = batch_annotate_clusters(
    marker_genes_list=marker_genes_list,
    species='mouse',                      # Organism species
    provider='anthropic',                 # LLM provider
    model='claude-3-7-sonnet-20250219',    # Specific model
    tissue='brain'                       # Optional tissue context
)

# Process and utilize results
for i, annotations in enumerate(batch_annotations):
    print(f"Dataset {i+1} annotations:")
    for cluster, annotation in annotations.items():
        print(f"  Cluster {cluster}: {annotation}")
```

### Multi-LLM Consensus Annotation

```python
from mllmcelltype import interactive_consensus_annotation, print_consensus_summary

# Define marker genes for each cluster
marker_genes = {
    "1": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # T cells
    "2": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # B cells
    "3": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # Monocytes
}

# Run iterative consensus annotation with multiple LLMs
result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',                                      # Organism species
    tissue='peripheral blood',                            # Tissue context
    models=[                                              # Multiple LLM models
        'gpt-4o',                                         # OpenAI
        'claude-3-7-sonnet-20250219',                     # Anthropic
        'gemini-2.5-pro',                                 # Google
        'qwen-max-2025-01-25'                             # Alibaba
    ],
    consensus_threshold=0.7,                              # Agreement threshold
    max_discussion_rounds=3,                              # Iterative refinement
    verbose=True                                          # Detailed output
)

# Print comprehensive consensus summary with uncertainty metrics
print_consensus_summary(result)

# Access results programmatically
final_annotations = result["consensus"]
uncertainty_metrics = {
    "consensus_proportion": result["consensus_proportion"],  # Agreement level
    "entropy": result["entropy"]                            # Annotation uncertainty
}
```

### Model Performance Analysis

```python
from mllmcelltype import compare_model_predictions, create_comparison_table
import matplotlib.pyplot as plt
import seaborn as sns

# Compare results from different LLM providers
model_predictions = {
    "OpenAI (GPT-4o)": results_openai,
    "Anthropic (Claude 3.7)": results_claude,
    "Google (Gemini 2.5)": results_gemini,
    "Alibaba (Qwen-Max)": results_qwen
}

# Perform comprehensive model comparison analysis
agreement_df, metrics = compare_model_predictions(
    model_predictions=model_predictions,
    display_plot=False                # We'll customize the visualization
)

# Generate detailed performance metrics
print(f"Average inter-model agreement: {metrics['agreement_avg']:.2f}")
print(f"Agreement variance: {metrics['agreement_var']:.2f}")
if 'accuracy' in metrics:
    print(f"Average accuracy: {metrics['accuracy_avg']:.2f}")

# Create custom visualization of model agreement patterns
plt.figure(figsize=(10, 8))
sns.heatmap(agreement_df, annot=True, cmap='viridis', vmin=0, vmax=1)
plt.title('Inter-model Agreement Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('model_agreement.png', dpi=300)
plt.show()

# Create and display a comparison table
comparison_table = create_comparison_table(model_predictions)
print(comparison_table)
```

### Custom Prompt Templates

```python
from mllmcelltype import annotate_clusters

# Define specialized prompt template for improved annotation precision
custom_template = """You are an expert computational biologist specializing in single-cell RNA-seq analysis.
Please annotate the following cell clusters based on their marker gene expression profiles.

Organism: {context}

Differentially expressed genes by cluster:
{clusters}

For each cluster, provide a precise cell type annotation based on canonical markers.
Consider developmental stage, activation state, and lineage information when applicable.
Provide only the cell type name for each cluster, one per line.
"""

# Annotate with specialized custom prompt
annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',                # Organism species
    provider='openai',              # LLM provider
    model='gpt-4o',                # Specific model
    prompt_template=custom_template # Custom instruction template
)
```

### Structured JSON Response Format

mLLMCelltype supports structured JSON responses, providing detailed annotation information with confidence scores and supporting evidence:

```python
from mllmcelltype import annotate_clusters

# Define comprehensive JSON response template
json_template = """
 You are an expert single-cell genomics analyst. Below are marker genes for different cell clusters from {context} tissue.

{clusters}

For each numbered cluster, provide a detailed cell type annotation in JSON format.
Use the following structure:

{
  "annotations": [
    {
      "cluster": "1",
      "cell_type": "precise cell type name",
      "confidence": "high/medium/low",
      "key_markers": ["marker1", "marker2", "marker3"],
      "evidence": "Brief explanation of key markers supporting this annotation",
      "alternative_annotation": "possible alternative if confidence is not high"
    }
  ]
}
"""

# Generate structured annotations with detailed metadata
json_annotations = annotate_clusters(
    marker_genes=marker_genes_df,
    species='human',                # Organism species
    tissue='lung',                  # Tissue context
    provider='openai',              # LLM provider
    model='gpt-4o',                # Specific model
    prompt_template=json_template   # JSON response template
)

# The parser automatically extracts structured data from the JSON response
for cluster_id, annotation in json_annotations.items():
    cell_type = annotation['cell_type']
    confidence = annotation['confidence']
    key_markers = ', '.join(annotation['key_markers'])
    print(f"Cluster {cluster_id}: {cell_type} (Confidence: {confidence})")
    print(f"  Key markers: {key_markers}")
    if 'evidence' in annotation:
        print(f"  Evidence: {annotation['evidence']}")

# Raw JSON response is also available in the cache for advanced processing
```

Using JSON responses provides several advantages:
- Structured data that can be easily processed
- Additional metadata like confidence levels and key markers
- More consistent parsing across different LLM providers

## Contributing

We welcome contributions to mLLMCelltype! Please feel free to submit issues or pull requests on our [GitHub repository](https://github.com/cafferychen777/mLLMCelltype).

## License

MIT License

## Citation

If you use mLLMCelltype in your research, please cite:

```bibtex
@software{mllmcelltype2025,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {mLLMCelltype: An iterative multi-LLM consensus framework for cell type annotation},
  url = {https://github.com/cafferychen777/mLLMCelltype},
  version = {1.0.0},
  year = {2025}
}
```

## Acknowledgements

We thank the developers of the various LLM APIs that make this framework possible, and the single-cell community for valuable feedback during development.
