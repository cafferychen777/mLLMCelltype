<div align="center">
  <img src="assets/mLLMCelltype_logo.png" alt="mLLMCelltype - Multi-LLM Consensus Framework for Cell Type Annotation in scRNA-seq" width="300"/>
</div>

<div align="center">
  <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>
</div>

<div align="center">
  <a href="https://twitter.com/intent/tweet?text=Check%20out%20mLLMCelltype%3A%20A%20multi-LLM%20consensus%20framework%20for%20cell%20type%20annotation%20in%20scRNA-seq%20data%21&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype"><img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype" alt="Tweet"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="Stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="Forks"></a>
</div>

<div align="center">
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="Issues">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="Release">
</div>

# mLLMCelltype: Multi-LLM Consensus Framework for Cell Type Annotation

mLLMCelltype is an advanced, iterative multi-LLM consensus framework designed for accurate and reliable cell type annotation in single-cell RNA sequencing (scRNA-seq) data. By leveraging the collective intelligence of multiple large language models including OpenAI GPT-4o/4.1, Anthropic Claude-3.7/3.5, Google Gemini-2.0, X.AI Grok-3, DeepSeek-V3, Alibaba Qwen2.5, Zhipu GLM-4, MiniMax, Stepfun, and OpenRouter, this framework significantly improves annotation accuracy while providing transparent uncertainty quantification for bioinformatics and computational biology research.

## Abstract

mLLMCelltype represents a significant advancement in computational methods for single-cell transcriptomics analysis. This open-source software leverages multiple large language models in a structured deliberation process to accurately identify cell types from gene expression data. Unlike traditional annotation methods that rely on reference datasets or single AI models, mLLMCelltype implements a novel consensus approach that reduces hallucinations, improves accuracy, and provides transparent uncertainty metrics. The framework is designed for seamless integration with popular single-cell analysis platforms like Scanpy and Seurat, making it accessible to researchers across the bioinformatics community.

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

<div align="center">
  <a href="https://star-history.com/#cafferychen777/mLLMCelltype">
    <img src="https://api.star-history.com/svg?repos=cafferychen777/mLLMCelltype&type=Date" alt="Star History Chart" width="600"/>
  </a>
  <p><em>Figure 1: mLLMCelltype GitHub star history showing community adoption and growth over time.</em></p>
</div>

## News

🎉 **April 2025**: We're thrilled to announce that less than a week after our preprint release, mLLMCelltype has surpassed 100 GitHub stars! We've also seen tremendous coverage from various media outlets and content creators. We extend our heartfelt gratitude to everyone who has supported this project through stars, shares, and contributions. Your enthusiasm drives our continued development and improvement of mLLMCelltype.

## Key Features

- **Multi-LLM Consensus Architecture**: Harnesses collective intelligence from diverse LLMs to overcome single-model limitations and biases
- **Structured Deliberation Process**: Enables LLMs to share reasoning, evaluate evidence, and refine annotations through multiple rounds of collaborative discussion
- **Transparent Uncertainty Quantification**: Provides quantitative metrics (Consensus Proportion and Shannon Entropy) to identify ambiguous cell populations requiring expert review
- **Hallucination Reduction**: Cross-model deliberation actively suppresses inaccurate or unsupported predictions through critical evaluation
- **Robust to Input Noise**: Maintains high accuracy even with imperfect marker gene lists through collective error correction
- **Hierarchical Annotation Support**: Optional extension for multi-resolution analysis with parent-child consistency
- **No Reference Dataset Required**: Performs accurate annotation without pre-training or reference data
- **Complete Reasoning Chains**: Documents the full deliberation process for transparent decision-making
- **Seamless Integration**: Works directly with standard Scanpy/Seurat workflows and marker gene outputs
- **Modular Design**: Easily incorporate new LLMs as they become available

## Recent Updates

### v1.1.4 (2025-04-24)

#### Bug Fixes
- Fixed a critical issue with cluster index handling, now the package strictly accepts only 0-based indices (compatible with Seurat)
- Fixed negative index (-1) issues that could occur when processing CSV input files
- Added strict validation for input cluster indices to ensure they start from 0

#### Improvements
- Removed automatic conversion logic from 1-based to 0-based indices in `prompt_templates.R`
- Added input validation in `consensus_annotation.R` to ensure cluster indices start from 0
- Updated code comments to clearly indicate that the package only accepts 0-based indices

### v1.1.3 (2025-04-15)
- Added support for X.AI's Grok models
- Updated the list of supported models, including Gemini 2.5 Pro
- Improved error handling and logging

See [NEWS.md](R/NEWS.md) for a complete changelog.

## Directory Structure

- `R/`: R language interface and implementation
- `python/`: Python interface and implementation

## Installation

### R Version

```r
# Install from GitHub
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")
```

### Python Version

```bash
# Install from PyPI
pip install mllmcelltype

# Or install from GitHub
pip install git+https://github.com/cafferychen777/mLLMCelltype.git
```

### Supported Models

- **OpenAI**: GPT-4.1/GPT-4.5/GPT-4o ([API Key](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([API Key](https://console.anthropic.com/))
- **Google**: Gemini-2.0-Pro/Gemini-2.0-Flash ([API Key](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([API Key](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-V3/DeepSeek-R1 ([API Key](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([API Key](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([API Key](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([API Key](https://bigmodel.cn/))
- **X.AI**: Grok-3/Grok-3-mini ([API Key](https://accounts.x.ai/))
- **OpenRouter**: Access to multiple models through a single API ([API Key](https://openrouter.ai/keys))
  - Supports models from OpenAI, Anthropic, Meta, Google, Mistral, and more
  - Format: 'provider/model-name' (e.g., 'openai/gpt-4o', 'anthropic/claude-3-opus')

## Usage Examples

### Python

```python
# Example of using mLLMCelltype for single-cell RNA-seq cell type annotation with Scanpy
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, setup_logging, interactive_consensus_annotation
import os

# Initialize logging for mLLMCelltype framework
setup_logging()

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

# Configure API keys for the large language models used in consensus annotation
# At least one API key is required for multi-LLM consensus annotation
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"      # For GPT-4o/4.1 models (recommended)
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # For Claude-3.7/3.5 models
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"      # For Google Gemini-2.0 models
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"        # For Alibaba Qwen2.5 models
# Additional optional LLM providers for enhanced consensus diversity:
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"   # For DeepSeek-V3 models
# os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"       # For Zhipu GLM-4 models
# os.environ["STEPFUN_API_KEY"] = "your-stepfun-api-key"    # For Stepfun models
# os.environ["MINIMAX_API_KEY"] = "your-minimax-api-key"    # For MiniMax models

# Execute multi-LLM consensus cell type annotation with iterative deliberation
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,  # Dictionary of marker genes for each cluster
    species="human",            # Specify organism for appropriate cell type annotation
    tissue="blood",            # Specify tissue context for more accurate annotation
    models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-1.5-pro", "qwen-max-2025-01-25"],  # Multiple LLMs for consensus
    consensus_threshold=1,     # Minimum proportion required for consensus agreement
    max_discussion_rounds=3    # Number of deliberation rounds between models for refinement
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

### R

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
# - OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: Access to models from multiple providers through a single API. Format: 'provider/model-name'
#   - OpenAI models: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic models: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#   - Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google models: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# Run LLMCelltype annotation with multiple LLM models
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # provide tissue context
  models = c(
    "claude-3-7-sonnet-20250219",  # Anthropic
    "gpt-4o",                   # OpenAI
    "gemini-1.5-pro",           # Google
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
# Use seurat_clusters to match each cell with its corresponding cluster metrics
current_clusters <- pbmc$seurat_clusters
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

# Create cache and log directories
cache_dir <- "path/to/your/cache"
log_dir <- "path/to/your/logs"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

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

# Run consensus annotation
consensus_results <- 
  interactive_consensus_annotation(
    input = marker_genes_list,
    tissue_name = "your tissue type", # e.g., "human heart"
    models = c("gemini-2.0-flash", 
              "gemini-1.5-pro", 
              "qwen-max-2025-01-25", 
              "grok-3-latest", 
              "anthropic/claude-3-7-sonnet-20250219",
              "openai/gpt-4o"),
    api_keys = api_keys,
    controversy_threshold = 0.6,
    entropy_threshold = 1.0,
    max_discussion_rounds = 3,
    cache_dir = cache_dir,
    log_dir = log_dir
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
# - OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
# - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
# - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
# - Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
# - Qwen: 'qwen-max-2025-01-25'
# - Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
# - Zhipu: 'glm-4-plus', 'glm-3-turbo'
# - MiniMax: 'minimax-text-01'
# - Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
# - OpenRouter: Access to models from multiple providers through a single API. Format: 'provider/model-name'
#   - OpenAI models: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
#   - Anthropic models: 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-5-sonnet-latest', 'anthropic/claude-3-5-haiku-latest', 'anthropic/claude-3-opus'
#   - Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
#   - Google models: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
#   - Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
#   - Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

# Run cell type annotation with a single LLM model
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # provide tissue context
  model = "claude-3-7-sonnet-20250219",  # specify a single model
  api_key = "your-anthropic-key",  # provide the API key directly
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
  "claude-3-7-sonnet-20250219",  # Anthropic
  "gpt-4o",                      # OpenAI
  "gemini-1.5-pro",              # Google
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

## Visualization Example

Below is an example of publication-ready visualization created with mLLMCelltype and SCpubr, showing cell type annotations alongside uncertainty metrics (Consensus Proportion and Shannon Entropy):

<div align="center">
  <img src="images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>
</div>

*Figure: Left panel shows cell type annotations on UMAP projection. Middle panel displays the consensus proportion using a yellow-green-blue gradient (deeper blue indicates stronger agreement among LLMs). Right panel shows Shannon entropy using an orange-red gradient (deeper red indicates lower uncertainty, lighter orange indicates higher uncertainty).*

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

We welcome and appreciate contributions from the community! There are many ways you can contribute to mLLMCelltype:

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

Join our [Discord community](https://discord.gg/pb2aZdG4) to get real-time updates about mLLMCelltype, ask questions, share your experiences, or collaborate with other users and developers. This is a great place to connect with the team and other users working on single-cell RNA-seq analysis.

Thank you for helping improve mLLMCelltype!
