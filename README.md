<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype logo" width="300"/>
</div>

<div align="center">
  <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="GitHub forks"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://www.r-pkg.org/badges/version/mLLMCelltype" alt="CRAN version"></a>
  <a href="https://CRAN.R-project.org/package=mLLMCelltype"><img src="https://cranlogs.r-pkg.org/badges/grand-total/mLLMCelltype" alt="CRAN downloads"></a>
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv preprint"></a>
  <a href="https://pypi.org/project/mllmcelltype/"><img src="https://img.shields.io/pypi/v/mllmcelltype" alt="PyPI version"></a>
  <a href="https://colab.research.google.com/github/cafferychen777/mLLMCelltype/blob/main/notebooks/mLLMCelltype_Tutorial.ipynb"><img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Open in Colab"></a>
</div>

# mLLMCelltype: Multi-LLM Consensus Framework for Cell Type Annotation

mLLMCelltype is a multi-LLM consensus framework for automated cell type annotation in single-cell RNA sequencing (scRNA-seq) data. The framework integrates multiple large language models including OpenAI GPT-5/4.1, Anthropic Claude-4/3.7/3.5, Google Gemini-2.0, X.AI Grok-3, DeepSeek-V3, Alibaba Qwen2.5, Zhipu GLM-4, MiniMax, Stepfun, and OpenRouter to improve annotation accuracy through consensus-based predictions.

## Abstract

mLLMCelltype is an open-source tool for single-cell transcriptomics analysis that uses multiple large language models to identify cell types from gene expression data. The software implements a consensus approach where multiple models analyze the same data and their predictions are combined, which helps reduce errors and provides uncertainty metrics. This methodology offers advantages over single-model approaches through integration of multiple model predictions. mLLMCelltype integrates with single-cell analysis platforms such as Scanpy and Seurat, allowing researchers to incorporate it into existing workflows. The method does not require reference datasets for annotation.

In our benchmarks ([Yang et al., 2025](https://doi.org/10.1101/2025.04.10.647852)), the consensus approach achieved up to 95% accuracy on tested datasets.

## Table of Contents
- [News](#news)
- [Key Features](#key-features)
- [Recent Updates](#recent-updates)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Visualization Example](#visualization-example)
- [Citation](#citation)
- [Contributing](#contributing)

## News

**Web Application Launch (2025-06-18)**

The mLLMCelltype web application is now available. You can access cell type annotation functionality directly through your web browser without any installation required.

**Features:**
- Upload your scRNA-seq data and get annotations
- Choose from various AI models including GPT-4, Claude, Gemini, and more
- Monitor annotation progress with live updates
- Download results in CSV, TSV, Excel, or JSON formats
- No setup required

**Access the Web App**: [https://mllmcelltype.com](https://mllmcelltype.com)

**Beta Testing Phase**: The web application is currently in beta testing. We welcome feedback and suggestions through our [GitHub Issues](https://github.com/cafferychen777/mLLMCelltype/issues) or [Discord community](https://discord.gg/pb2aZdG4).

**CRAN Release (2025-09-02)**

mLLMCelltype is now available on CRAN. Install the package using:
```R
install.packages("mLLMCelltype")
```

- **CRAN page**: https://CRAN.R-project.org/package=mLLMCelltype
- **DOI**: 10.32614/CRAN.package.mLLMCelltype

**Important: Gemini Model Migration (2025-06-02)**

Google has discontinued several Gemini 1.5 models and will discontinue more on September 24, 2025:
- **Already discontinued**: Gemini 1.5 Pro 001, Gemini 1.5 Flash 001
- **Will be discontinued on Sept 24, 2025**: Gemini 1.5 Pro 002, Gemini 1.5 Flash 002, Gemini 1.5 Flash-8B -001

**Recommended migration**: Use `gemini-2.0-flash` or `gemini-2.0-flash-lite` for continued support. The aliases `gemini-1.5-pro` and `gemini-1.5-flash` will continue to work until September 24, 2025, as they point to the -002 versions.

**Important: Claude Model Deprecation (2025-07-21)**

Anthropic will retire the following Claude models on July 21, 2025:
- **Claude 2** (all versions)
- **Claude 2.1**
- **Claude 3 Sonnet** (non-versioned)
- **Claude 3 Opus** (non-versioned)

**Recommended migration**:
- For Claude 2/2.1 → Use `claude-sonnet-4-5-20250929` (latest)
- For Claude 3 Sonnet → Use `claude-sonnet-4-5-20250929` (latest)
- For Claude 3 Opus → Use `claude-opus-4-1-20250805` (latest)

Please update your code before July 21, 2025 to avoid service disruption.

See also: [FlashDeconv](https://github.com/cafferychen777/FlashDeconv) for spatial transcriptomics deconvolution.

## Key Features

- **Multi-LLM Consensus**: Integrates predictions from multiple LLMs to reduce single-model limitations and biases
- **Model Support**: Compatible with 10+ LLM providers including OpenAI, Anthropic, Google, and others
- **Iterative Discussion**: LLMs evaluate evidence and refine annotations through multiple rounds of discussion
- **Uncertainty Quantification**: Provides Consensus Proportion and Shannon Entropy metrics to identify uncertain annotations
- **Cross-Model Validation**: Reduces incorrect predictions through multi-model comparison
- **Noise Tolerance**: Maintains accuracy with imperfect marker gene lists
- **Hierarchical Annotation**: Supports multi-resolution analysis with consistency checks
- **Reference-Free**: Performs annotation without pre-training or reference datasets
- **Documentation**: Records complete reasoning process for transparency
- **Integration**: Compatible with Scanpy/Seurat workflows and marker gene outputs
- **Extensibility**: Supports addition of new LLMs as they become available

## Recent Updates

### v1.2.3 (2025-05-10)

#### Bug Fixes
- Fixed error handling in consensus checking when API responses are NULL or invalid
- Improved error logging for OpenRouter API error responses
- Added robust NULL and type checking in check_consensus function

#### Improvements
- Enhanced error diagnostics for OpenRouter API errors
- Added detailed logging of API error messages and response structures
- Improved robustness when handling unexpected API response formats

### v1.2.2 (2025-05-09)

#### Bug Fixes
- Fixed the 'non-character argument' error that occurred when processing API responses
- Added robust type checking for API responses across all model providers
- Improved error handling for unexpected API response formats

#### Improvements
- Added detailed error logging for API response issues
- Implemented consistent error handling patterns across all API processing functions
- Enhanced response validation to ensure proper structure before processing

### v1.2.1 (2025-05-01)

#### Improvements
- Added support for OpenRouter API
- Added support for free models through OpenRouter
- Updated documentation with examples for using OpenRouter models

### v1.2.0 (2025-04-30)

#### Features
- Added visualization functions for cell type annotation results
- Added support for uncertainty metrics visualization
- Implemented improved consensus building algorithm

### v1.1.5 (2025-04-27)

#### Bug Fixes
- Fixed an issue with cluster index validation that caused errors when processing certain CSV input files
- Improved error handling for negative indices with clearer error messages

#### Improvements
- Added example script for CSV-based annotation workflow (cat_heart_annotation.R)
- Enhanced input validation with more detailed diagnostics
- Updated documentation to clarify CSV input format requirements

See [NEWS.md](R/NEWS.md) for a complete changelog.

## Directory Structure

- `R/`: R language interface and implementation
- `python/`: Python interface and implementation

## Installation

### R Version

```r
# Install from CRAN (recommended)
install.packages("mLLMCelltype")

# Or install development version from GitHub
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Python Version

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZgmtlaORogSy0-QsaF0CHwFWOyOD26d2?usp=sharing)

**Quick Start**: Try mLLMCelltype in Google Colab without any installation. Click the badge above to open an interactive notebook with examples and step-by-step guidance.

```bash
# Install from PyPI
pip install mllmcelltype

# Or install from GitHub (note the subdirectory parameter)
pip install git+https://github.com/cafferychen777/mLLMCelltype.git#subdirectory=python
```

#### Important Note on Dependencies

mLLMCelltype uses a modular design where different LLM provider libraries are optional dependencies. Depending on which models you plan to use, you'll need to install the corresponding packages:

```bash
# For using OpenAI models (GPT-5, etc.)
pip install "mllmcelltype[openai]"

# For using Anthropic models (Claude)
pip install "mllmcelltype[anthropic]"

# For using Google models (Gemini)
pip install "mllmcelltype[gemini]"

# To install all optional dependencies at once
pip install "mllmcelltype[all]"
```

If you encounter errors like `ImportError: cannot import name 'genai' from 'google'`, it means you need to install the corresponding provider package. For example:

```bash
# For Google Gemini models
pip install google-genai
```

### Supported Models

- **OpenAI**: GPT-5/GPT-4.1/GPT-4.5 ([API Key](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-4-Opus/Claude-4-Sonnet/Claude-3.7-Sonnet/Claude-3.5-Haiku ([API Key](https://console.anthropic.com/))
- **Google**: Gemini-2.0-Pro/Gemini-2.0-Flash ([API Key](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([API Key](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([API Key](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([API Key](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([API Key](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([API Key](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([API Key](https://accounts.x.ai/))
- **OpenRouter**: Access to multiple models through a single API ([API Key](https://openrouter.ai/keys))
  - Supports models from OpenAI, Anthropic, Meta, Google, Mistral, and more
  - Format: 'provider/model-name' (e.g., 'openai/gpt-5', 'anthropic/claude-opus-4.1')
  - Free models available with `:free` suffix (e.g., 'deepseek/deepseek-r1:free', 'meta-llama/llama-4-maverick:free')
  - **Note**: Free tier limits (as of Oct 2025): 50 requests/day (1000/day with $10+ credits), 20 requests/minute. Some models may be unavailable.

## Usage Examples

### Python

```python
# Example of using mLLMCelltype for single-cell RNA-seq cell type annotation with Scanpy
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, interactive_consensus_annotation
import os

# Note: Logging is automatically configured when importing mllmcelltype
# You can customize logging if needed using the logging module

# Load your single-cell RNA-seq dataset in AnnData format
adata = sc.read_h5ad('your_data.h5ad')  # Replace with your scRNA-seq dataset path

# Perform Leiden clustering for cell population identification if not already done
if 'leiden' not in adata.obs.columns:
    print("Computing leiden clustering for cell population identification...")
    # Preprocess single-cell data: normalize counts and log-transform for gene expression analysis
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize to 10,000 counts per cell
        sc.pp.log1p(adata)  # Log-transform normalized counts

    # Dimensionality reduction: calculate PCA for scRNA-seq data
    if 'X_pca' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)  # Select informative genes
        sc.pp.pca(adata, use_highly_variable=True)  # Compute principal components

    # Cell clustering: compute neighborhood graph and perform Leiden community detection
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)  # Build KNN graph for clustering
    sc.tl.leiden(adata, resolution=0.8)  # Identify cell populations using Leiden algorithm
    print(f"Leiden clustering completed, identified {len(adata.obs['leiden'].cat.categories)} distinct cell populations")

# Identify marker genes for each cell cluster using differential expression analysis
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')  # Wilcoxon rank-sum test for marker detection

# Extract top marker genes for each cell cluster to use in cell type annotation
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # Select top 10 differentially expressed genes as markers for each cluster
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# IMPORTANT: mLLMCelltype requires gene symbols (e.g., KCNJ8, PDGFRA) not Ensembl IDs (e.g., ENSG00000176771)
# If your AnnData object uses Ensembl IDs, convert them to gene symbols for accurate annotation:
# Example conversion code:
# if 'Gene' in adata.var.columns:  # Check if gene symbols are available in the metadata
#     gene_name_dict = dict(zip(adata.var_names, adata.var['Gene']))
#     marker_genes = {cluster: [gene_name_dict.get(gene_id, gene_id) for gene_id in genes]
#                    for cluster, genes in marker_genes.items()}

# IMPORTANT: mLLMCelltype requires numeric cluster IDs
# The 'cluster' column must contain numeric values or values that can be converted to numeric.
# Non-numeric cluster IDs (e.g., "cluster_1", "T_cells", "7_0") may cause errors or unexpected behavior.
# If your data contains non-numeric cluster IDs, create a mapping between original IDs and numeric IDs:
# Example standardization code:
# original_ids = list(marker_genes.keys())
# id_mapping = {original: idx for idx, original in enumerate(original_ids)}
# marker_genes = {str(id_mapping[cluster]): genes for cluster, genes in marker_genes.items()}

# Configure API keys for the large language models used in consensus annotation
# At least one API key is required for multi-LLM consensus annotation
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"      # For GPT-5/4.1 models (recommended)
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # For Claude-4/3.7/3.5 models
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"      # For Google Gemini-2.5 models
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"        # For Alibaba Qwen2.5 models
# Additional optional LLM providers for enhanced consensus diversity:
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"   # For DeepSeek-V3 models
# os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"       # For Zhipu GLM-4 models
# os.environ["STEPFUN_API_KEY"] = "your-stepfun-api-key"    # For Stepfun models
# os.environ["MINIMAX_API_KEY"] = "your-minimax-api-key"    # For MiniMax models
# os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"  # For accessing multiple models via OpenRouter

# Execute multi-LLM consensus cell type annotation with iterative deliberation
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,  # Dictionary of marker genes for each cluster
    species="human",            # Specify organism for appropriate cell type annotation
    tissue="blood",            # Specify tissue context for more accurate annotation
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro", "qwen-max-2025-01-25"],  # Multiple LLMs for consensus
    consensus_threshold=1,     # Minimum proportion required for consensus agreement
    max_discussion_rounds=3    # Number of deliberation rounds between models for refinement
)

# Alternatively, use OpenRouter for accessing multiple models through a single API
# This is especially useful for accessing free models with the :free suffix
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Example using free OpenRouter models (no credits required)
free_models_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},      # Meta Llama 4 Maverick (free)
        {"provider": "openrouter", "model": "venice/uncensored:free"},                # Venice Uncensored (free)
        {"provider": "openrouter", "model": "deepseek/deepseek-r1:free"},             # DeepSeek R1 (free, advanced reasoning)
        {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"} # Meta Llama 3.3 70B (free)
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=2
)

# Retrieve final consensus cell type annotations from the multi-LLM deliberation
final_annotations = consensus_results["consensus"]

# Integrate consensus cell type annotations into the original AnnData object
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)

# Add uncertainty quantification metrics to evaluate annotation confidence
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])  # Agreement level
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])  # Annotation uncertainty

# Prepare for visualization: compute UMAP embeddings if not already available
# UMAP provides a 2D representation of cell populations for visualization
if 'X_umap' not in adata.obsm:
    print("Computing UMAP coordinates...")
    # Make sure neighbors are computed first
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    print("UMAP coordinates computed")

# Visualize results with enhanced aesthetics
# Basic visualization
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='right', frameon=True, title='mLLMCelltype Consensus Annotations')

# More customized visualization
import matplotlib.pyplot as plt

# Set figure size and style
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Create a more publication-ready UMAP
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='on data',
         frameon=True, title='mLLMCelltype Consensus Annotations',
         palette='tab20', size=50, legend_fontsize=12,
         legend_fontoutline=2, ax=ax)

# Visualize uncertainty metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
sc.pl.umap(adata, color='consensus_proportion', ax=ax1, title='Consensus Proportion',
         cmap='viridis', vmin=0, vmax=1, size=30)
sc.pl.umap(adata, color='entropy', ax=ax2, title='Annotation Uncertainty (Shannon Entropy)',
         cmap='magma', vmin=0, size=30)
plt.tight_layout()
```

### Using a Single Free OpenRouter Model

For users who prefer a simpler approach with just one model, the DeepSeek R1 free model via OpenRouter can be used without API credits:

```python
import os
from mllmcelltype import annotate_clusters

# Note: Logging is automatically configured

# Set your OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Define marker genes for each cluster
marker_genes = {
    "0": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7"],           # T cells
    "1": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74"],   # B cells
    "2": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A"]      # Monocytes
}

# Annotate using DeepSeek R1 free model
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # Free model with advanced reasoning
)

# Print annotations
for cluster, annotation in annotations.items():
    print(f"Cluster {cluster}: {annotation}")
```

This approach uses a free model and does not require API credits.

#### Extracting Marker Genes from AnnData Objects

If you're using Scanpy with AnnData objects, you can easily extract marker genes directly from the `rank_genes_groups` results:

```python
import os
import scanpy as sc
from mllmcelltype import annotate_clusters

# Note: Logging is automatically configured

# Set your OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"

# Load and preprocess your data
adata = sc.read_h5ad('your_data.h5ad')

# Perform preprocessing and clustering if not already done
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata)
# sc.pp.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.leiden(adata)

# Find marker genes for each cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Extract top marker genes for each cluster
marker_genes = {
    cluster: adata.uns['rank_genes_groups']['names'][cluster][:10].tolist()
    for cluster in adata.obs['leiden'].cat.categories
}

# Annotate using DeepSeek R1 free model
annotations = annotate_clusters(
    marker_genes=marker_genes,
    species='human',
    tissue='peripheral blood',  # adjust based on your tissue type
    provider='openrouter',
    model='deepseek/deepseek-r1:free'  # Free model
)

# Add annotations to AnnData object
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(annotations)

# Visualize results
sc.pl.umap(adata, color='cell_type', legend_loc='on data',
           frameon=True, title='Cell Types Annotated by DeepSeek R1')
```

This method automatically extracts the top differentially expressed genes for each cluster from the `rank_genes_groups` results, making it easy to integrate mLLMCelltype into your Scanpy workflow.

### R

> **Note**: For more detailed R tutorials and documentation, please visit the [mLLMCelltype documentation website](https://cafferyang.com/mLLMCelltype/).

#### Using Seurat Object

```r
# Load required packages
library(mLLMCelltype)
library(Seurat)
library(dplyr)
library(ggplot2)
library(cowplot) # Added for plot_grid

# Load your preprocessed Seurat object
pbmc <- readRDS("your_seurat_object.rds")

# If starting with raw data, perform preprocessing steps
# pbmc <- NormalizeData(pbmc)
# pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
# pbmc <- ScaleData(pbmc)
# pbmc <- RunPCA(pbmc)
# pbmc <- FindNeighbors(pbmc, dims = 1:10)
# pbmc <- FindClusters(pbmc, resolution = 0.5)
# pbmc <- RunUMAP(pbmc, dims = 1:10)

# Find marker genes for each cluster
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# Set up cache directory to speed up processing
cache_dir <- "./mllmcelltype_cache"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

# Choose a model from any supported provider
# Supported models include:
# - OpenAI: 'gpt-5', 'gpt-5-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-sonnet-4-5-20250929', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: Access to models from multiple providers through a single API. Format: 'provider/model-name'
#   - OpenAI models: 'openai/gpt-5', 'openai/gpt-5-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic models: 'anthropic/claude-sonnet-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4', 'anthropic/claude-opus-4.1'
#   - Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google models: 'google/gemini-3-pro', 'google/gemini-3-flash', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# Run LLMCelltype annotation with multiple LLM models
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # provide tissue context
  models = c(
    "claude-sonnet-4-5-20250929",    # Anthropic Claude 4 (latest)
    "gpt-5",                    # OpenAI
    "gemini-3-pro",           # Google
    "qwen-max-2025-01-25"       # Alibaba
  ),
  api_keys = list(
    anthropic = "your-anthropic-key",
    openai = "your-openai-key",
    gemini = "your-google-key",
    qwen = "your-qwen-key"
  ),
  top_gene_count = 10,
  controversy_threshold = 1.0,
  entropy_threshold = 1.0,
  cache_dir = cache_dir
)

# Print structure of results to understand the data
print("Available fields in consensus_results:")
print(names(consensus_results))

# Add annotations to Seurat object
# Get cell type annotations from consensus_results$final_annotations
cluster_to_celltype_map <- consensus_results$final_annotations

# Create new cell type identifier column
cell_types <- as.character(Idents(pbmc))
for (cluster_id in names(cluster_to_celltype_map)) {
  cell_types[cell_types == cluster_id] <- cluster_to_celltype_map[[cluster_id]]
}

# Add cell type annotations to Seurat object
pbmc$cell_type <- cell_types

# Add uncertainty metrics
# Extract detailed consensus results containing metrics
consensus_details <- consensus_results$initial_results$consensus_results

# Create a data frame with metrics for each cluster
uncertainty_metrics <- data.frame(
  cluster_id = names(consensus_details),
  consensus_proportion = sapply(consensus_details, function(res) res$consensus_proportion),
  entropy = sapply(consensus_details, function(res) res$entropy)
)

# Add uncertainty metrics for each cell
# Note: seurat_clusters is a metadata column automatically created by FindClusters() function
# It contains the cluster ID assigned to each cell during clustering
# Here we use it to map cluster-level metrics (consensus_proportion and entropy) to individual cells

# If you don't have seurat_clusters column (e.g., if you used a different clustering method),
# you can use the active identity (Idents) or any other cluster assignment in your metadata:
# Option 1: Use active identity
# current_clusters <- as.character(Idents(pbmc))
# Option 2: Use another metadata column that contains cluster IDs
# current_clusters <- pbmc$your_cluster_column

# For this example, we use the standard seurat_clusters column:
current_clusters <- pbmc$seurat_clusters  # Get cluster ID for each cell

# Match each cell's cluster ID with the corresponding metrics in uncertainty_metrics
pbmc$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(current_clusters, uncertainty_metrics$cluster_id)]
pbmc$entropy <- uncertainty_metrics$entropy[match(current_clusters, uncertainty_metrics$cluster_id)]

# Save results for future use
saveRDS(consensus_results, "pbmc_mLLMCelltype_results.rds")
saveRDS(pbmc, "pbmc_annotated.rds")

# Visualize results with SCpubr for publication-ready plots
if (!requireNamespace("SCpubr", quietly = TRUE)) {
  remotes::install_github("enblacar/SCpubr")
}
library(SCpubr)
library(viridis)  # For color palettes

# Basic UMAP visualization with default settings
pdf("pbmc_basic_annotations.pdf", width=8, height=6)
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  legend.position = "right") +
  ggtitle("mLLMCelltype Consensus Annotations")
dev.off()

# More customized visualization with enhanced styling
pdf("pbmc_custom_annotations.pdf", width=8, height=6)
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  label.box = TRUE,
                  legend.position = "right",
                  pt.size = 1.0,
                  border.size = 1,
                  font.size = 12) +
  ggtitle("mLLMCelltype Consensus Annotations") +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

# Visualize uncertainty metrics with enhanced SCpubr plots
# Get cell types and create a named color palette
cell_types <- unique(pbmc$cell_type)
color_palette <- viridis::viridis(length(cell_types))
names(color_palette) <- cell_types

# Cell type annotations with SCpubr
p1 <- SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  legend.position = "bottom",  # Place legend at the bottom
                  pt.size = 1.0,
                  label.size = 4,  # Smaller label font size
                  label.box = TRUE,  # Add background box to labels for better readability
                  repel = TRUE,  # Make labels repel each other to avoid overlap
                  colors.use = color_palette,
                  plot.title = "Cell Type") +
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            legend.text = element_text(size = 8),
            legend.key.size = unit(0.3, "cm"),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Consensus proportion feature plot with SCpubr
p2 <- SCpubr::do_FeaturePlot(sample = pbmc,
                       features = "consensus_proportion",
                       order = TRUE,
                       pt.size = 1.0,
                       enforce_symmetry = FALSE,
                       legend.title = "Consensus",
                       plot.title = "Consensus Proportion",
                       sequential.palette = "YlGnBu",  # Yellow-Green-Blue gradient, following Nature Methods standards
                       sequential.direction = 1,  # Light to dark direction
                       min.cutoff = min(pbmc$consensus_proportion),  # Set minimum value
                       max.cutoff = max(pbmc$consensus_proportion),  # Set maximum value
                       na.value = "lightgrey") +  # Color for missing values
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Shannon entropy feature plot with SCpubr
p3 <- SCpubr::do_FeaturePlot(sample = pbmc,
                       features = "entropy",
                       order = TRUE,
                       pt.size = 1.0,
                       enforce_symmetry = FALSE,
                       legend.title = "Entropy",
                       plot.title = "Shannon Entropy",
                       sequential.palette = "OrRd",  # Orange-Red gradient, following Nature Methods standards
                       sequential.direction = -1,  # Dark to light direction (reversed)
                       min.cutoff = min(pbmc$entropy),  # Set minimum value
                       max.cutoff = max(pbmc$entropy),  # Set maximum value
                       na.value = "lightgrey") +  # Color for missing values
      theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
            plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Combine plots with equal widths
pdf("pbmc_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
```

#### Using CSV Input

You can also use mLLMCelltype with CSV files directly without Seurat, which is useful for cases where you already have marker genes available in CSV format:

```r
# Install the latest version of mLLMCelltype
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R", force = TRUE)

# Load necessary packages
library(mLLMCelltype)

# Configure unified logging (optional - uses defaults if not specified)
configure_logger(level = "INFO", console_output = TRUE, json_format = TRUE)

# Create cache directory
cache_dir <- "path/to/your/cache"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

# Read CSV file content
markers_file <- "path/to/your/markers.csv"
file_content <- readLines(markers_file)

# Skip header row
data_lines <- file_content[-1]

# Convert data to list format, using numeric indices as keys
marker_genes_list <- list()
cluster_names <- c()

# First collect all cluster names
for(line in data_lines) {
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  cluster_names <- c(cluster_names, parts[1])
}

# Then create marker_genes_list with numeric indices
for(i in 1:length(data_lines)) {
  line <- data_lines[i]
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]

  # First part is the cluster name
  cluster_name <- parts[1]

  # Use index as key (0-based index, compatible with Seurat)
  cluster_id <- as.character(i - 1)

  # Remaining parts are genes
  genes <- parts[-1]

  # Filter out NA and empty strings
  genes <- genes[!is.na(genes) & genes != ""]

  # Add to marker_genes_list
  marker_genes_list[[cluster_id]] <- list(genes = genes)
}

# Set API keys
api_keys <- list(
  gemini = "YOUR_GEMINI_API_KEY",
  qwen = "YOUR_QWEN_API_KEY",
  grok = "YOUR_GROK_API_KEY",
  openai = "YOUR_OPENAI_API_KEY",
  anthropic = "YOUR_ANTHROPIC_API_KEY"
)

# Run consensus annotation with paid models
consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # e.g., "human heart"
    models = c("gemini-3-pro",
              "gemini-3-flash",
              "qwen-max-2025-01-25",
              "grok-3-latest",
              "claude-sonnet-4-5-20250929",
              "gpt-5"),
    api_keys = api_keys,
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 3,
    cache_dir = cache_dir
  )

# Alternatively, use free OpenRouter models (no credits required)
# Add OpenRouter API key to the api_keys list
api_keys$openrouter <- "your-openrouter-api-key"

# Run consensus annotation with free models
free_consensus_results <-
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # e.g., "human heart"
    models = c(
      "meta-llama/llama-4-maverick:free",      # Meta Llama 4 Maverick (free)
      "venice/uncensored:free",                # Venice Uncensored (free)
      "deepseek/deepseek-r1:free",             # DeepSeek R1 (free, advanced reasoning)
      "meta-llama/llama-3.3-70b-instruct:free" # Meta Llama 3.3 70B (free)
    ),
    api_keys = api_keys,
    consensus_check_model = "deepseek/deepseek-r1:free",  # Free model for consensus checking
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 2,
    cache_dir = cache_dir
  )

# Save results
saveRDS(consensus_results, "your_results.rds")

# Print results summary
cat("\nResults summary:\n")
cat("Available fields:", paste(names(consensus_results), collapse=", "), "\n\n")

# Print final annotations
cat("Final cell type annotations:\n")
for(cluster in names(consensus_results$final_annotations)) {
  cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
}
```

**Notes on CSV format**:
- The CSV file should have values in the first column that will be used as indices (these can be cluster names, numbers like 0,1,2,3 or 1,2,3,4, etc.)
- The values in the first column are only used for reference and are not passed to the LLMs
- Subsequent columns should contain marker genes for each cluster
- An example CSV file for cat heart tissue is included in the package at `inst/extdata/Cat_Heart_markers.csv`

Example CSV structure:
```
cluster,gene
0,Negr1,Cask,Tshz2,Ston2,Fstl1,Dse,Celf2,Hmcn2,Setbp1,Cblb
1,Palld,Grb14,Mybpc3,Ensfcag00000044939,Dcun1d2,Acacb,Slco1c1,Ppp1r3c,Sema3c,Ppp1r14c
2,Adgrf5,Tbx1,Slco2b1,Pi15,Adam23,Bmx,Pde8b,Pkhd1l1,Dtx1,Ensfcag00000051556
3,Clec2d,Trat1,Rasgrp1,Card11,Cytip,Sytl3,Tmem156,Bcl11b,Lcp1,Lcp2
```

You can access the example data in your R script using:
```r
system.file("extdata", "Cat_Heart_markers.csv", package = "mLLMCelltype")
```

### Using a Single LLM Model

If you only want to use a single LLM model instead of the consensus approach, use the `annotate_cell_types()` function. This is useful when you have access to only one API key or prefer a specific model:

```r
# Load required packages
library(mLLMCelltype)
library(Seurat)

# Load your preprocessed Seurat object
pbmc <- readRDS("your_seurat_object.rds")

# Find marker genes for each cluster
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

# Choose a model from any supported provider
# Supported models include:
# - OpenAI: 'gpt-5', 'gpt-5-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-sonnet-4-5-20250929', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: Access to models from multiple providers through a single API. Format: 'provider/model-name'
#   - OpenAI models: 'openai/gpt-5', 'openai/gpt-5-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic models: 'anthropic/claude-sonnet-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4', 'anthropic/claude-opus-4.1'
#   - Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google models: 'google/gemini-3-pro', 'google/gemini-3-flash', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# Run cell type annotation with a single LLM model
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # provide tissue context
  model = "claude-sonnet-4-5-20250929",  # specify a single model (Claude 4 Opus)
  api_key = "your-anthropic-key",  # provide the API key directly
  top_gene_count = 10
)

# Using a free OpenRouter model
free_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "meta-llama/llama-4-maverick:free",  # free model with :free suffix
  api_key = "your-openrouter-key",
  top_gene_count = 10
)

# Print the results
print(single_model_results)

# Add annotations to Seurat object
# single_model_results is a character vector with one annotation per cluster
pbmc$cell_type <- plyr::mapvalues(
  x = as.character(Idents(pbmc)),
  from = as.character(0:(length(single_model_results)-1)),
  to = single_model_results
)

# Visualize results
DimPlot(pbmc, group.by = "cell_type", label = TRUE) +
  ggtitle("Cell Types Annotated by Single LLM Model")
```

#### Comparing Different Models

You can also compare annotations from different models by running `annotate_cell_types()` multiple times with different models:

```r
# Define models to test
models_to_test <- c(
  "claude-sonnet-4-5-20250929",     # Anthropic
  "gpt-5",                       # OpenAI
  "gemini-3-pro",              # Google
  "qwen-max-2025-01-25"          # Alibaba
)

# API keys for different providers
api_keys <- list(
  anthropic = "your-anthropic-key",
  openai = "your-openai-key",
  gemini = "your-gemini-key",
  qwen = "your-qwen-key"
)

# Test each model and store results
results <- list()
for (model in models_to_test) {
  provider <- get_provider(model)
  api_key <- api_keys[[provider]]

  # Run annotation
  results[[model]] <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = model,
    api_key = api_key,
    top_gene_count = 10
  )

  # Add to Seurat object
  column_name <- paste0("cell_type_", gsub("[^a-zA-Z0-9]", "_", model))
  pbmc[[column_name]] <- plyr::mapvalues(
    x = as.character(Idents(pbmc)),
    from = as.character(0:(length(results[[model]])-1)),
    to = results[[model]]
  )
}
```

### Advanced Consensus Configuration: Specifying the Consensus Check Model

The `consensus_check_model` parameter (R) / `consensus_model` parameter (Python) allows you to specify which LLM model to use for consensus checking and discussion moderation. This parameter is important for the accuracy of consensus annotation because the consensus check model:

1. Evaluates semantic similarity between different cell type annotations
2. Calculates consensus metrics (proportion and entropy)
3. Moderates and synthesizes discussions between models for controversial clusters
4. Makes final decisions when models disagree

We recommend using a capable model for consensus checking, as this directly impacts annotation quality.

#### Recommended Models for Consensus Checking

- **Anthropic**: `claude-sonnet-4-5-20250929`, `claude-opus-4-1-20250805`
- **OpenAI**: `o1`, `o1-pro`, `gpt-5`, `gpt-4.1`
- **Google**: `gemini-3-pro`, `gemini-3-flash`, `gemini-2.0-flash`
- **Other**: `deepseek-r1` / `deepseek-reasoner`, `qwen-max-2025-01-25`, `grok-3-latest`

#### R Package Usage

```r
# Example 1: Specifying a consensus check model
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human brain",
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro", "qwen-max-2025-01-25"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# Example 2: Using an alternative consensus check model
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "mouse liver",
  models = c("gpt-5", "gemini-3-pro", "qwen-max-2025-01-25"),
  api_keys = api_keys,
  consensus_check_model = "claude-sonnet-4-5-20250929",
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)

# Example 3: Using OpenAI's reasoning model
consensus_results <- interactive_consensus_annotation(
  input = marker_genes_list,
  tissue_name = "human immune cells",
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro"),
  api_keys = api_keys,
  consensus_check_model = "o1",
  controversy_threshold = 0.7,
  entropy_threshold = 1.0
)
```

#### Python Package Usage

```python
# Example 1: Specifying a consensus model
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="brain",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro", "qwen-max-2025-01-25"],
    consensus_model="claude-sonnet-4-5-20250929",
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Example 2: Using dictionary format
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="mouse",
    tissue="liver",
    models=["gpt-5", "gemini-3-pro", "qwen-max-2025-01-25"],
    consensus_model={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Example 3: Using Google's model for consensus
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="heart",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "qwen-max-2025-01-25"],
    consensus_model={"provider": "google", "model": "gemini-3-pro"},
    consensus_threshold=0.7,
    entropy_threshold=1.0
)

# Example 4: Default behavior (uses Qwen with fallback)
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro"],
    # If not specified, defaults to qwen-max-2025-01-25 with claude-sonnet-4-5-20250929 as fallback
    consensus_threshold=0.7,
    entropy_threshold=1.0
)
```

#### Notes on Consensus Model Selection

1. **Model Availability**: Ensure you have API access to your chosen consensus model. The system will use fallback models if the primary choice is unavailable.

2. **Consistency**: Use the same model for all consensus checks within a project to ensure consistent evaluation criteria.

3. **Default Behavior**:
   - R: Uses the first model in the `models` list if not specified
   - Python: Defaults to `qwen-max-2025-01-25` with `claude-sonnet-4-5-20250929` as fallback

The consensus check model must accurately assess semantic similarity between different cell type names (e.g., recognizing that "T lymphocyte" and "T cell" refer to the same cell type), understand biological context, and synthesize discussions from multiple models.

### Advanced Features: Cluster Selection and Cache Control (v1.3.1)

mLLMCelltype v1.3.1 introduces two parameters that give you fine-grained control over the annotation process:

#### 1. **clusters_to_analyze** - Selective Cluster Analysis

This parameter allows you to specify exactly which clusters to analyze without manually filtering your input data:

```r
# Example: Focus on specific clusters for T cell subtyping
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC - T cell subtypes",
  models = c("gpt-5", "claude-sonnet-4-5-20250929"),
  api_keys = api_keys,
  clusters_to_analyze = c(0, 1, 7),  # Only analyze T cell clusters
  controversy_threshold = 0.7
)

# Example: Re-analyze controversial clusters with different context
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "activated immune cells",
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro"),
  api_keys = api_keys,
  clusters_to_analyze = c("3", "5"),  # Focus on specific clusters
  cache_dir = "consensus_cache"
)
```

**Benefits:**
- No need to subset your data manually
- Maintains original cluster numbering
- Reduces API calls and costs by only analyzing relevant clusters
- Useful for iterative refinement of specific cell populations

#### 2. **force_rerun** - Bypass Cache for Fresh Analysis

This parameter forces re-analysis of controversial clusters, bypassing cached results:

```r
# Example: Initial broad analysis
initial_results <- interactive_consensus_annotation(
  input = markers,
  tissue_name = "human brain",
  models = c("gpt-5", "claude-sonnet-4-5-20250929"),
  api_keys = api_keys,
  use_cache = TRUE
)

# Example: Re-analyze with specific subtype context
subtype_results <- interactive_consensus_annotation(
  input = markers,
  tissue_name = "human brain - neuronal subtypes",
  models = c("gpt-5", "claude-sonnet-4-5-20250929"),
  api_keys = api_keys,
  clusters_to_analyze = c(2, 3, 5),  # Neuronal clusters
  force_rerun = TRUE,  # Force fresh analysis despite cache
  use_cache = TRUE     # Still benefit from cache for non-controversial clusters
)
```

**Important Notes:**
- `force_rerun` only affects controversial clusters requiring LLM discussion
- Non-controversial clusters still use cache for performance
- Useful when changing tissue context or focusing on subtypes
- Combines well with `clusters_to_analyze` for targeted re-analysis

#### Common Use Cases

1. **Iterative Subtyping Workflow:**
```r
# Step 1: General cell type annotation
general_types <- interactive_consensus_annotation(
  input = data,
  tissue_name = "human PBMC",
  models = models,
  api_keys = api_keys
)

# Step 2: Focus on T cells with subtype context
t_cell_subtypes <- interactive_consensus_annotation(
  input = data,
  tissue_name = "human T lymphocytes",
  models = models,
  api_keys = api_keys,
  clusters_to_analyze = c(0, 1, 4, 7),  # T cell clusters from step 1
  force_rerun = TRUE  # Fresh analysis with T cell context
)

# Step 3: Further refine CD8+ T cells
cd8_subtypes <- interactive_consensus_annotation(
  input = data,
  tissue_name = "human CD8+ T cells - activation states",
  models = models,
  api_keys = api_keys,
  clusters_to_analyze = c(1, 4),  # CD8+ clusters
  force_rerun = TRUE
)
```

2. **Cost-Effective Re-analysis:**
```r
# Only re-analyze clusters that were controversial
controversial <- initial_results$controversial_clusters

refined_results <- interactive_consensus_annotation(
  input = data,
  tissue_name = "human PBMC - refined",
  models = c("gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-pro"),
  api_keys = api_keys,
  clusters_to_analyze = controversial,  # Only controversial ones
  force_rerun = TRUE,
  consensus_check_model = "claude-sonnet-4-5-20250929"
)
```

## Visualization Examples

### Cell Type Annotation Visualization

Below is an example of publication-ready visualization created with mLLMCelltype and SCpubr, showing cell type annotations alongside uncertainty metrics (Consensus Proportion and Shannon Entropy):

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*Figure: Left panel shows cell type annotations on UMAP projection. Middle panel displays the consensus proportion using a yellow-green-blue gradient (deeper blue indicates stronger agreement among LLMs). Right panel shows Shannon entropy using an orange-red gradient (deeper red indicates lower uncertainty, lighter orange indicates higher uncertainty).*

### Marker Gene Visualization

mLLMCelltype includes marker gene visualization functions that integrate with the consensus annotation workflow:

```r
# Load required libraries
library(mLLMCelltype)
library(Seurat)
library(ggplot2)

# After running consensus annotation
consensus_results <- interactive_consensus_annotation(
  input = markers_df,
  tissue_name = "human PBMC",
  models = c("anthropic/claude-sonnet-4.5", "openai/gpt-5"),
  api_keys = list(openrouter = "your_api_key")
)

# Create marker gene visualizations using Seurat
# Add consensus annotations to Seurat object
cluster_ids <- as.character(Idents(pbmc_data))
cell_type_annotations <- consensus_results$final_annotations[cluster_ids]

# Handle any missing annotations
if (any(is.na(cell_type_annotations))) {
  na_mask <- is.na(cell_type_annotations)
  cell_type_annotations[na_mask] <- paste("Cluster", cluster_ids[na_mask])
}

# Add to Seurat object
pbmc_data@meta.data$cell_type_consensus <- cell_type_annotations

# Create a dotplot of marker genes
DotPlot(pbmc_data,
        features = top_markers,
        group.by = "cell_type_consensus") +
  RotatedAxis()

# Create a heatmap of marker genes
DoHeatmap(pbmc_data,
          features = top_markers,
          group.by = "cell_type_consensus")
```

**Marker Gene Visualization Features:**

- **DotPlot**: Shows both percentage of cells expressing each gene (dot size) and average expression level (color intensity)
- **Heatmap**: Displays scaled expression values with clustering of genes and cell types
- **Integration**: Works directly with consensus annotation results added to Seurat objects
- **Standard Seurat Functions**: Uses familiar Seurat visualization functions for consistency

For detailed instructions and advanced customization options, see the [Visualization Guide](R/vignettes/06-visualization-guide.html).

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

You can also cite this in plain text format:

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. [Read our full research paper on bioRxiv](https://doi.org/10.1101/2025.04.10.647852)

## Contributing

We welcome contributions from the community. There are many ways you can contribute to mLLMCelltype:

### Reporting Issues

If you encounter any bugs, have feature requests, or have questions about using mLLMCelltype, please [open an issue](https://github.com/cafferychen777/mLLMCelltype/issues) on our GitHub repository. When reporting bugs, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your operating system and package version information
- Any relevant code snippets or error messages

### Pull Requests

We encourage you to contribute code improvements or new features through pull requests:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

Here are some areas where contributions would be particularly valuable:

- Adding support for new LLM models
- Improving documentation and examples
- Optimizing performance
- Adding new visualization options
- Extending functionality for specialized cell types or tissues
- Translations of documentation into different languages

### Code Style

Please follow the existing code style in the repository. For R code, we generally follow the [tidyverse style guide](https://style.tidyverse.org/). For Python code, we follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).

### Community

Join our [Discord community](https://discord.gg/pb2aZdG4) for discussion and questions about mLLMCelltype and single-cell RNA-seq analysis.

Thank you for helping improve mLLMCelltype!
