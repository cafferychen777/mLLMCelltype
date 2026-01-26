# mLLMCelltype

[![PyPI version](https://img.shields.io/badge/pypi-v1.1.0-blue.svg)](https://pypi.org/project/mllmcelltype/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

mLLMCelltype is a comprehensive Python framework for automated cell type annotation in single-cell RNA sequencing data through an iterative multi-LLM consensus approach. By leveraging the collective intelligence of multiple large language models, this framework significantly improves annotation accuracy while providing robust uncertainty quantification. The package is fully compatible with the scverse ecosystem, allowing seamless integration with AnnData objects and Scanpy workflows.

### Scientific Background

Single-cell RNA sequencing has revolutionized our understanding of cellular heterogeneity, but accurate cell type annotation remains challenging. Traditional annotation methods often rely on reference datasets or manual expert curation, which can be time-consuming and subjective. mLLMCelltype addresses these limitations by implementing a novel multi-model deliberative framework that:

1. Harnesses complementary strengths of diverse LLMs to overcome single-model limitations
2. Implements a structured deliberation process for collaborative reasoning
3. Provides quantitative uncertainty metrics to identify ambiguous annotations
4. Maintains high accuracy even with imperfect marker gene inputs

## Key Features

### Multi-LLM Architecture
- **Comprehensive Provider Support**:
  - OpenAI (GPT-5, O1, etc.)
  - Anthropic (Claude 4 Opus, Claude 4 Sonnet, Claude 4 Sonnet, Claude 3.5 Haiku, etc.)
  - Google (Gemini 3 Pro, Gemini 3 Flash, Gemini 2.5 Pro, etc.)
  - Alibaba (Qwen-Max-2025-01-25, Qwen-Plus, etc.)
  - DeepSeek (DeepSeek-Chat, DeepSeek-Reasoner)
  - StepFun (Step-2-16k, Step-2-Mini, Step-1-Flash, Step-1-8k, Step-1-32k, etc.)
  - Zhipu AI (GLM-4, GLM-3-Turbo)
  - MiniMax (MiniMax-Text-01)
  - X.AI (Grok-3-latest)
  - OpenRouter (Access to multiple models through a single API)
    - Supports models from OpenAI, Anthropic, Meta, Mistral and more
    - Format: 'provider/model-name' (e.g., 'openai/gpt-5', 'anthropic/claude-3-opus')
    - Free models available with `:free` suffix (e.g., 'meta-llama/llama-4-maverick:free')
    - Free tier limits (as of Oct 2025): 50 requests/day (1000/day with $10+ credits)
- **Seamless Integration**:
  - Works directly with Scanpy/AnnData workflows
  - Compatible with scverse ecosystem
  - Flexible input formats (dictionary, DataFrame, or AnnData)

### Advanced Annotation Capabilities
- **Iterative Consensus Framework**: Enables multiple rounds of structured deliberation between LLMs
- **Uncertainty Quantification**: Provides Consensus Proportion (CP) and Shannon Entropy (H) metrics
- **Hallucination Reduction**: Cross-model verification minimizes unsupported predictions
- **Hierarchical Annotation**: Optional support for multi-resolution analysis with parent-child consistency

### Technical Features
- **Unified API**: Consistent interface across all LLM providers
- **Custom Base URL Support**: Configure custom API endpoints for proxy servers, enterprise gateways, or alternative endpoints
- **Smart Endpoint Selection**: Automatic fallback between international and domestic endpoints (e.g., Qwen)
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
    model='gpt-5',               # Specific model
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
export OPENAI_API_KEY="your-openai-api-key"  # For GPT models
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # For Claude models
export GOOGLE_API_KEY="your-google-api-key"  # For Gemini models
export QWEN_API_KEY="your-qwen-api-key"  # For Qwen-Max-2025-01-25, Qwen-Plus
export DEEPSEEK_API_KEY="your-deepseek-api-key"  # For DeepSeek-Chat
export ZHIPU_API_KEY="your-zhipu-api-key"  # For GLM-4, GLM-3-Turbo
export STEPFUN_API_KEY="your-stepfun-api-key"  # For Step-2-16k, Step-2-Mini, etc.
export MINIMAX_API_KEY="your-minimax-api-key"  # For MiniMax-Text-01
export GROK_API_KEY="your-grok-api-key"  # For Grok-3-latest
export OPENROUTER_API_KEY="your-openrouter-api-key"  # For accessing multiple models via OpenRouter
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

## Custom Base URL Configuration

mLLMCelltype v1.3.0+ supports custom base URLs for API endpoints, enabling usage with proxy servers, enterprise API gateways, or alternative endpoints. This is particularly useful for users in regions with API access restrictions or organizations with custom API infrastructure.

### Single Base URL for All Providers

```python
from mllmcelltype import annotate_clusters

# Use a single proxy URL for all providers
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    provider='openai',
    model='gpt-5',
    api_key='your-api-key',
    base_urls='https://your-proxy.com/v1/chat/completions'  # Single URL for all
)
```

### Provider-Specific Base URLs

```python
from mllmcelltype import interactive_consensus_annotation

# Configure different base URLs for different providers
base_urls = {
    'openai': 'https://openai-proxy.com/v1/chat/completions',
    'anthropic': 'https://anthropic-proxy.com/v1/messages',
    'qwen': 'https://qwen-proxy.com/compatible-mode/v1/chat/completions'
    # Other providers will use default endpoints
}

consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    models=['gpt-5', 'claude-3-opus', 'qwen-max'],
    api_keys={
        'openai': 'your-openai-key',
        'anthropic': 'your-anthropic-key',
        'qwen': 'your-qwen-key'
    },
    base_urls=base_urls
)
```

### Smart Endpoint Selection (Qwen)

For Qwen models, mLLMCelltype automatically tests endpoint connectivity and selects the best available endpoint:

```python
# Qwen automatically selects between international and domestic endpoints
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    provider='qwen',
    model='qwen-max-2025-01-25',
    api_key='your-qwen-key'
    # No base_urls needed - automatic endpoint selection
)
```

### Configuration for Chinese Users

Chinese users can benefit from custom base URLs to access international APIs:

```python
# Recommended configuration for Chinese users
china_base_urls = {
    # International APIs through proxy
    'openai': 'https://your-openai-proxy.com/v1/chat/completions',
    'anthropic': 'https://your-anthropic-proxy.com/v1/messages',
    'gemini': 'https://your-gemini-proxy.com/v1beta/models',

    # Domestic APIs use default endpoints (no proxy needed)
    # 'qwen', 'deepseek', 'zhipu' automatically use optimal endpoints
}

consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    models=[
        'gpt-5',           # Through proxy
        'claude-3-opus',    # Through proxy
        'qwen-max',         # Direct access with smart endpoint selection
        'deepseek-chat',    # Direct access
        'glm-4-plus'        # Direct access
    ],
    api_keys=your_api_keys,
    base_urls=china_base_urls
)
```

## Advanced Usage

### Targeted Analysis: New Enhanced Parameters

mLLMCelltype v1.3.0+ introduces two powerful parameters for more precise control over cell type annotation:

#### `clusters_to_analyze`: Focus on Specific Clusters

This parameter allows you to analyze only specific clusters of interest, rather than processing all clusters in your dataset.

```python
from mllmcelltype import interactive_consensus_annotation

# Example: Large dataset with 10 clusters, but only interested in immune cells
all_marker_genes = {
    "cluster_0": ["CD3D", "CD3E", "CD3G", "IL7R", "TCF7"],        # T cells
    "cluster_1": ["CD79A", "CD79B", "MS4A1", "IGHM", "IGKC"],     # B cells
    "cluster_2": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1"],     # Monocytes  
    "cluster_3": ["FCGR3A", "NCR1", "KLRD1", "GNLY", "PRF1"],     # NK cells
    "cluster_4": ["EPCAM", "KRT8", "KRT18", "KRT19"],             # Epithelial cells
    "cluster_5": ["COL1A1", "COL3A1", "FN1", "VIM"],             # Fibroblasts
    "cluster_6": ["PECAM1", "VWF", "ENG", "CDH5"],               # Endothelial cells
    "cluster_7": ["PPBP", "PF4", "TUBB1", "GP9", "ITGA2B"],      # Platelets
    "cluster_8": ["HBB", "HBA1", "HBA2", "ALAS2"],               # Erythrocytes
    "cluster_9": ["TPSAB1", "TPSB2", "CPA3", "MS4A2"]           # Mast cells
}

# Focus analysis only on immune cells (clusters 0, 1, 2, 3)
result = interactive_consensus_annotation(
    marker_genes=all_marker_genes,
    species="human",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro"],
    clusters_to_analyze=["cluster_0", "cluster_1", "cluster_2", "cluster_3"],  # Only immune clusters
    tissue="peripheral blood",
    verbose=True
)

print("Analyzed clusters:", list(result['consensus'].keys()))
# Output: ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3']
```

**Key Benefits:**
- **Efficiency**: Save time and API costs by focusing on clusters of interest
- **Targeted Analysis**: Perfect for specific research questions (e.g., "What are the immune cell subtypes?")
- **Iterative Workflow**: Analyze different subsets in separate runs for different research objectives

#### `force_rerun`: Fresh Analysis Bypassing Cache

This parameter forces fresh analysis by ignoring cached results, useful for re-analyzing clusters with different contexts or parameters.

```python
# Scenario: Previous analysis might have been with limited context
# Now you want fresh analysis with more specific context

marker_genes = {
    "cluster_0": ["CD3D", "CD3E", "CD8A", "PRF1", "GZMB"],  # Could be CD8+ T cells or NK cells
    "cluster_1": ["CD19", "MS4A1", "CD79A", "IGHM", "CD27"] # B cells with memory markers
}

# First analysis with general context
print("=== First Analysis (will be cached) ===")
result1 = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    models=["gpt-5", "claude-sonnet-4-5-20250929"],
    tissue="peripheral blood",  # General context
    force_rerun=False,  # Use cache if available (default)
    verbose=False
)

# Second analysis with specific disease context - force fresh analysis
print("=== Second Analysis (force fresh analysis) ===")  
result2 = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human", 
    models=["gpt-5", "claude-sonnet-4-5-20250929"],
    tissue="peripheral blood",
    additional_context="Patient with autoimmune disease, focus on activated/memory cell states",
    force_rerun=True,  # Force fresh analysis, ignore cache
    verbose=False
)

print("Cached result:", result1['consensus'])
print("Fresh result:", result2['consensus'])
```

**Key Benefits:**
- **Context Refinement**: Re-analyze with better biological context or updated knowledge
- **Parameter Exploration**: Test different consensus thresholds or model combinations
- **Quality Control**: Verify consistent results across multiple runs

#### Combined Usage: Maximum Flexibility

Both parameters can be used together for ultimate control:

```python
# Scenario: You have a large dataset but want to re-analyze specific controversial clusters

controversial_clusters = ["cluster_5", "cluster_8", "cluster_12"]  # Previously identified as uncertain

result = interactive_consensus_annotation(
    marker_genes=all_marker_genes,
    species="human",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro", "qwen-max-2025-01-25"],
    clusters_to_analyze=controversial_clusters,  # Focus on specific clusters
    force_rerun=True,                           # Force fresh analysis
    tissue="brain",
    additional_context="Focus on rare neuronal subtypes and glial cell distinctions",
    consensus_threshold=0.8,  # Higher threshold for more stringent consensus
    max_discussion_rounds=3,
    verbose=True
)

print(f"Re-analyzed {len(controversial_clusters)} controversial clusters with fresh context")
```

**Use Cases:**
- **Iterative Refinement**: Progressively improve annotations for challenging clusters
- **Publication Preparation**: Ensure robust, well-contextualized annotations for important findings
- **Method Comparison**: Compare different model combinations on the same clusters
- **Quality Assurance**: Validate critical cell type assignments with multiple approaches

### Using OpenRouter

OpenRouter provides a unified API for accessing models from multiple providers. Our comprehensive testing shows that OpenRouter integration works seamlessly in all scenarios, including complex cell types and multi-round discussions.

#### Single Model Annotation

```python
from mllmcelltype import annotate_clusters

# Set your OpenRouter API key
import os
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Define marker genes for each cluster
marker_genes = {
    "1": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # T cells
    "2": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # B cells
    "3": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # Monocytes
}

# Annotate using OpenAI's GPT-5 via OpenRouter
openai_annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider_config={"provider": "openrouter", "model": "openai/gpt-5"}
)

# Annotate using Anthropic's Claude model via OpenRouter
anthropic_annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider_config={"provider": "openrouter", "model": "anthropic/claude-3-opus"}
)

# Annotate using Meta's Llama model via OpenRouter
meta_annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider_config={"provider": "openrouter", "model": "meta-llama/llama-3-70b-instruct"}
)

# Annotate using a free model via OpenRouter
free_model_annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider_config={"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"}  # Free model with :free suffix
)

# Print annotations from different models
for cluster in marker_genes.keys():
    print(f"Cluster {cluster}:")
    print(f"  OpenAI GPT-5: {openai_annotations[cluster]}")
    print(f"  Anthropic Claude: {anthropic_annotations[cluster]}")
    print(f"  Meta Llama: {meta_annotations[cluster]}")
    print(f"  DeepSeek (free): {free_model_annotations[cluster]}")
```

#### Pure OpenRouter Consensus

You can run consensus annotation using only OpenRouter models. **Note: When using OpenRouter, you must specify models using a dictionary format with provider and model keys:**

```python
from mllmcelltype import interactive_consensus_annotation, print_consensus_summary

# Run consensus annotation with only OpenRouter models
result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    models=[
        {"provider": "openrouter", "model": "openai/gpt-5"},             # OpenRouter OpenAI (paid)
        {"provider": "openrouter", "model": "anthropic/claude-3-opus"},   # OpenRouter Anthropic (paid)
        {"provider": "openrouter", "model": "meta-llama/llama-3-70b-instruct"}  # OpenRouter Meta (paid)
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=3,
    verbose=True
)

# Print consensus summary
print_consensus_summary(result)
```

#### Using Free OpenRouter Models

OpenRouter provides access to free models with the `:free` suffix. These models don't require credits but may have limitations:

```python
from mllmcelltype import interactive_consensus_annotation, print_consensus_summary

# Run consensus annotation with free OpenRouter models
result = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},      # Llama 4 (free)
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},             # DeepSeek R1 (free)
        {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"}, # Llama 3.3 (free)
        {"provider": "openrouter", "model": "venice/uncensored:free"}                 # Venice (free)
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=3,
    verbose=True
)

# Print consensus summary
print_consensus_summary(result)
```

#### Using a Single Free OpenRouter Model

Based on user feedback, the Microsoft MAI-DS-R1 free model provides excellent results while being fast and accurate:

```python
from mllmcelltype import annotate_clusters, setup_logging

# Setup logging (optional)
setup_logging()

# Set your OpenRouter API key
import os
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Define marker genes for each cluster
marker_genes = {
    "0": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # T cells
    "1": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # B cells
    "2": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # Monocytes
}

# Annotate using only the Microsoft MAI-DS-R1 free model
mai_annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # Free model
)

# Print annotations
for cluster, annotation in mai_annotations.items():
    print(f"Cluster {cluster}: {annotation}")
```

This approach is particularly useful when:
- You need quick results without API costs
- You have limited API access to other providers
- You're performing initial exploratory analysis
- You want to validate results from other models

The Microsoft MAI-DS-R1 free model has shown excellent performance in cell type annotation tasks, often comparable to larger paid models.

**Note**: Free model availability may change over time. You can check the current list of available models on the OpenRouter website or through their API:

```python
import requests
import os

# Get your OpenRouter API key
api_key = os.environ.get("OPENROUTER_API_KEY", "your-openrouter-api-key")

# Get available models
response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

# Print all available models
models = response.json()["data"]
print("Available OpenRouter models:")
for model in models:
    model_id = model["id"]
    is_free = model.get("pricing", {}).get("prompt") == 0 and model.get("pricing", {}).get("completion") == 0
    print(f"  - {model_id}{' (free)' if is_free else ''}")
```

### Multi-LLM Consensus Annotation

#### Mixed Direct API and OpenRouter Models

Our testing confirms that OpenRouter models can seamlessly participate in consensus annotation alongside direct API models. They can also engage in discussion rounds when disagreements occur:

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
        'gpt-5',                                         # OpenAI direct API
        'claude-sonnet-4-5-20250929',                   # Anthropic direct API (latest)
        'gemini-3-pro',                                  # Google direct API
        'qwen-max-2025-01-25',                            # Alibaba direct API
        {"provider": "openrouter", "model": "openai/gpt-5"},             # OpenRouter (OpenAI)
        {"provider": "openrouter", "model": "anthropic/claude-3-opus"},   # OpenRouter (Anthropic)
        {"provider": "openrouter", "model": "meta-llama/llama-3-70b-instruct"}  # OpenRouter (Meta)
    ],
    consensus_threshold=0.7,                              # Agreement threshold
    max_discussion_rounds=3,                              # Iterative refinement
    verbose=True                                          # Detailed output
)

# Print comprehensive consensus summary with uncertainty metrics
print_consensus_summary(result)
```

#### Handling Complex Cell Types and Discussions

For challenging cell types that may trigger discussion rounds, OpenRouter models can effectively participate in the deliberation process:

```python
# For ambiguous or specialized cell types (e.g., regulatory T cells vs. CD4+ T cells)
result = interactive_consensus_annotation(
    marker_genes=specialized_marker_genes,  # Markers for specialized cell types
    species='human',
    tissue='lymphoid tissue',
    models=[
        'gpt-5',                                                      # Direct API (paid)
        {"provider": "openrouter", "model": "openai/gpt-5"},          # OpenRouter (paid)
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},    # OpenRouter (free)
    ],
    consensus_threshold=0.8,                # Higher threshold to force discussion
    max_discussion_rounds=3,                # Allow multiple rounds of discussion
    verbose=True
)

# Using only free models for budget-conscious users
result_free = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},      # Llama 4 (free)
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},             # DeepSeek R1 (free)
        {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"}, # Llama 3.3 (free)
        {"provider": "openrouter", "model": "venice/uncensored:free"}                 # Venice (free)
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=2,
    verbose=True
)
```

#### Manual Comparison of OpenRouter Models

You can also get individual annotations from different OpenRouter models and compare them manually:

```python
# Get annotations from different models via OpenRouter
openai_via_openrouter = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider_config={"provider": "openrouter", "model": "openai/gpt-5"}
)

anthropic_via_openrouter = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider_config={"provider": "openrouter", "model": "anthropic/claude-3-opus"}
)

# Create a dictionary of model predictions for comparison
model_predictions = {
    "OpenAI via OpenRouter": openai_via_openrouter,
    "Anthropic via OpenRouter": anthropic_via_openrouter,
    "Direct OpenAI": results_openai,  # From previous direct API calls
}

# Compare the results
from mllmcelltype import compare_model_predictions
agreement_df, metrics = compare_model_predictions(model_predictions)

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
    "OpenAI (GPT-5)": results_openai,
    "Anthropic (Claude 4)": results_claude,
    "Google (Gemini 2.5 Pro)": results_gemini,
    "Alibaba (Qwen-Max-2025-01-25)": results_qwen
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

### Advanced Consensus Configuration: Specifying the Consensus Model

The `consensus_model` parameter in `interactive_consensus_annotation` allows you to specify which LLM model to use for consensus checking and discussion moderation. This parameter is **critical** for annotation accuracy because the consensus model:

1. Evaluates semantic similarity between different cell type annotations
2. Calculates consensus metrics (proportion and entropy)
3. Moderates and synthesizes discussions between models for controversial clusters
4. Makes final decisions when models disagree

**⚠️ Important: We strongly recommend using the most capable models available for consensus checking, as this directly impacts annotation quality.**

#### Recommended Models for Consensus Checking (Ranked by Performance)

1. **Anthropic Claude Models** (Highest recommendation)
   - `claude-sonnet-4-5-20250929` - **Best overall** (latest, most intelligent)
   - `claude-opus-4-1-20250805` - Excellent for complex reasoning tasks

2. **OpenAI Models**
   - `o1` / `o1-pro` - Advanced reasoning capabilities
   - `gpt-5` - Strong performance across various cell types
   - `gpt-4.1` - Latest GPT-4 variant

3. **Google Gemini Models**
   - `gemini-3-pro` - Top-tier performance with enhanced reasoning
   - `gemini-3-flash` - Excellent balance of performance and speed
   - `gemini-2.0-flash` - Good performance with faster processing

4. **Other High-Performance Models**
   - `deepseek-r1` / `deepseek-reasoner` - Strong reasoning capabilities
   - `qwen-max-2025-01-25` - Excellent for scientific contexts (default)
   - `grok-3-latest` - Advanced language understanding

#### Usage Examples

```python
# Example 1: Using the best available model for consensus checking (Recommended)
from mllmcelltype import interactive_consensus_annotation

consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro", "qwen-max-2025-01-25"],
    consensus_model="claude-sonnet-4-5-20250929",  # Use the latest and most capable Sonnet model
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Example 2: Using dictionary format with a high-performance model
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="mouse",
    tissue="liver",
    models=["gpt-5", "gemini-3-pro", "qwen-max-2025-01-25"],
    consensus_model={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Example 3: Using Google's latest model for consensus
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="heart",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "qwen-max-2025-01-25"],
    consensus_model={"provider": "google", "model": "gemini-3-pro"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Example 4: Using OpenAI's reasoning model for complex cases
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="immune cells",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro"],
    consensus_model="o1",  # OpenAI's advanced reasoning model
    consensus_threshold=0.7,
    entropy_threshold=1.0,
    api_keys={"openai": "your-openai-api-key"}
)

# Example 5: Default behavior (uses high-performance Qwen model)
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro"],
    # If not specified, defaults to qwen-max-2025-01-25 with claude-sonnet-4-5-20250929 as fallback
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# ⚠️ NOT RECOMMENDED: Avoid using less capable or free models for consensus checking
# as this may significantly reduce annotation accuracy
```

#### Best Practices for Consensus Model Selection

1. **Prioritize Accuracy Over Cost**: The consensus model plays a crucial role in determining final annotations. Using a less capable model here can compromise the entire annotation process.

2. **Model Availability**: Ensure you have API access to your chosen consensus model. The system will use fallback models if the primary choice is unavailable:
   - Primary: Your specified model or `qwen-max-2025-01-25` (default)
   - Fallback: `claude-sonnet-4-5-20250929`

3. **Consistency**: Use the same high-performance model for all consensus checks within a project to ensure consistent evaluation criteria.

4. **Complex Tissues**: For challenging tissues (e.g., brain, immune system), consider using the most advanced models like Claude Opus, O1, or Gemini 2.5 Pro.

5. **API Key Management**: Make sure you have the appropriate API key for your chosen consensus model:
   ```python
   api_keys = {
       "openai": "your-openai-key",
       "anthropic": "your-anthropic-key",
       "google": "your-google-key",
       "qwen": "your-qwen-key"
   }
   ```

#### Why Model Quality Matters for Consensus Checking

The consensus model must:
- Accurately assess semantic similarity between different cell type names (e.g., recognizing that "T lymphocyte" and "T cell" refer to the same cell type)
- Understand biological context and hierarchical relationships
- Synthesize discussions from multiple models to reach accurate conclusions
- Provide reliable confidence metrics for downstream analysis

Using a less capable model for these critical tasks can lead to:
- Misidentification of controversial clusters
- Incorrect consensus calculations
- Poor resolution of disagreements between models
- Ultimately, less accurate cell type annotations

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
    model='gpt-5',                # Specific model
    prompt_template=custom_template # Custom instruction template
)
```

### Structured JSON Response Format

mLLMCelltype supports structured JSON responses, providing detailed annotation information with confidence scores and key markers:

```python
from mllmcelltype import annotate_clusters

# Define JSON response template matching the default implementation
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
      "key_markers": ["marker1", "marker2", "marker3"]
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
    model='gpt-5',                # Specific model
    prompt_template=json_template   # JSON response template
)

# The parser automatically extracts structured data from the JSON response
for cluster_id, annotation in json_annotations.items():
    cell_type = annotation['cell_type']
    confidence = annotation['confidence']
    key_markers = ', '.join(annotation['key_markers'])
    print(f"Cluster {cluster_id}: {cell_type} (Confidence: {confidence})")
    print(f"  Key markers: {key_markers}")

# Raw JSON response is also available in the cache for advanced processing
```

Using JSON responses provides several advantages:

- Structured data that can be easily processed
- Additional metadata like confidence levels and key markers
- More consistent parsing across different LLM providers

## Scanpy/AnnData Integration

mLLMCelltype is designed to seamlessly integrate with the scverse ecosystem, particularly with AnnData objects and Scanpy workflows.

### AnnData Integration

mLLMCelltype can directly process data from AnnData objects and add annotation results back to AnnData objects:

```python
import scanpy as sc
import mllmcelltype as mct

# Load data
adata = sc.datasets.pbmc3k()

# Preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)
sc.tl.umap(adata)

# Extract marker genes for each cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
marker_genes = {}
for cluster in adata.obs['leiden'].unique():
    genes = sc.get.rank_genes_groups_df(adata, group=cluster)['names'].tolist()[:20]
    marker_genes[cluster] = genes

# Use mLLMCelltype for cell type annotation
annotations = mct.annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    provider='openai',
    model='gpt-5'
)

# Add annotations back to AnnData object
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)

# Visualize results
sc.pl.umap(adata, color='cell_type', legend_loc='on data')
```

### Multi-Model Consensus Annotation with AnnData

mLLMCelltype's multi-model consensus framework also integrates seamlessly with AnnData:

```python
import mllmcelltype as mct

# Use multiple models for consensus annotation
consensus_results = mct.interactive_consensus_annotation(
    marker_genes=marker_genes,
    species='human',
    models=['gpt-5', 'claude-sonnet-4-5-20250929', 'gemini-3-pro', 'openai/gpt-5'],  # Can include OpenRouter models
    consensus_threshold=0.7
)

# Add consensus annotations and uncertainty metrics to AnnData object
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus"])
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])

# Visualize results
sc.pl.umap(adata, color=['consensus_cell_type', 'consensus_proportion', 'entropy'])
```

### Complete Scanpy Workflow Integration

Check our [examples directory](https://github.com/cafferychen777/mLLMCelltype/tree/main/python/examples) for complete Scanpy integration examples, including:

- scanpy_integration_example.py: Basic Scanpy workflow integration
- bcl_integration_example.py: Integration with Bioconductor/Seurat workflows
- discussion_mode_example.py: Advanced integration example using multi-model discussion mode

## Contributing

We welcome contributions to mLLMCelltype! Please feel free to submit issues or pull requests on our [GitHub repository](https://github.com/cafferychen777/mLLMCelltype).

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

## Acknowledgements

We thank the developers of the various LLM APIs that make this framework possible, and the single-cell community for valuable feedback during development.
