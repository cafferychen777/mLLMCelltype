# mLLMCelltype

[中文文档](README_CN.md)

mLLMCelltype is an iterative multi-LLM consensus framework for cell type annotation in single-cell RNA sequencing data. By leveraging the complementary strengths of multiple large language models (GPT, Claude, Gemini, Qwen, etc.), this framework significantly improves annotation accuracy while providing transparent uncertainty quantification.

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

- **OpenAI**: GPT-4.5/GPT-4o ([API Key](https://platform.openai.com/settings/organization/billing/overview))
- **Anthropic**: Claude-3.7-Sonnet/Claude-3.5-Haiku ([API Key](https://console.anthropic.com/))
- **Google**: Gemini-1.5-Pro/Gemini-1.5-Flash ([API Key](https://ai.google.dev/?authuser=2))
- **Alibaba**: Qwen2.5-Max ([API Key](https://www.alibabacloud.com/en/product/modelstudio))
- **DeepSeek**: DeepSeek-R1 ([API Key](https://platform.deepseek.com/usage))
- **Minimax**: MiniMax-Text-01 ([API Key](https://intl.minimaxi.com/user-center/basic-information/interface-key))
- **Stepfun**: Step-2-16K ([API Key](https://platform.stepfun.com/account-info))
- **Zhipu**: GLM-4 ([API Key](https://bigmodel.cn/))

## Usage Examples

### Python

```python
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, setup_logging, interactive_consensus_annotation
import os

# Set up logging
setup_logging()

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Run differential expression analysis to get marker genes
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Extract marker genes for each cluster
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # Extract top 10 genes for each cluster
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# Set API keys for different providers
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"
os.environ["QWEN_API_KEY"] = "your-qwen-api-key"

# Run consensus annotation with multiple models
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-1.5-pro", "qwen-max-2025-01-25"],
    consensus_threshold=0.7,  # Adjust threshold for consensus agreement
    max_discussion_rounds=3   # Maximum rounds of discussion between models
)

# Access the final consensus annotations from the dictionary
final_annotations = consensus_results["consensus"]

# Add consensus annotations to your AnnData object
adata.obs['consensus_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)

# Add uncertainty metrics to your AnnData object
adata.obs['consensus_proportion'] = adata.obs['leiden'].astype(str).map(consensus_results["consensus_proportion"])
adata.obs['entropy'] = adata.obs['leiden'].astype(str).map(consensus_results["entropy"])

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

```r
# Load required packages
library(mLLMCelltype)
library(Seurat)
library(dplyr)

# Load your preprocessed Seurat object
pbmc <- readRDS("your_seurat_object.rds")

# Find marker genes for each cluster
pbmc_markers <- FindAllMarkers(pbmc,
                            only.pos = TRUE,
                            min.pct = 0.25,
                            logfc.threshold = 0.25)

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
  controversy_threshold = 0.7
)

# Add annotations to Seurat object
# Create a mapping dictionary to correctly map cluster IDs to cell types
cluster_to_celltype_map <- setNames(
  unlist(consensus_results$final_annotations),
  names(consensus_results$final_annotations)
)

# Get current cluster IDs for each cell
current_clusters <- as.character(Idents(pbmc))

# Add cell type annotations to Seurat object
pbmc$cell_type <- cluster_to_celltype_map[current_clusters]

# Add uncertainty metrics
pbmc$consensus_proportion <- consensus_results$consensus_results[current_clusters]$consensus_proportion
pbmc$entropy <- consensus_results$consensus_results[current_clusters]$entropy

# Visualize results with SCpubr for publication-ready plots
if (!requireNamespace("SCpubr", quietly = TRUE)) {
  remotes::install_github("enblacar/SCpubr")
}
library(SCpubr)

# Basic UMAP visualization
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  legend.position = "right") +
  ggtitle("mLLMCelltype Consensus Annotations")

# For more customized visualization
SCpubr::do_DimPlot(sample = pbmc,
                  group.by = "cell_type",
                  label = TRUE,
                  label.box = FALSE,
                  legend.position = "right",
                  pt.size = 1.2,
                  border.size = 1,
                  font.size = 14) +
  ggtitle("mLLMCelltype Consensus Annotations")
```

## License

MIT

## Citation

If you use mLLMCelltype in your research, please cite:

```bibtex
@software{mllmcelltype2025,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {mLLMCelltype: An iterative multi-LLM consensus framework for cell type annotation},
  url = {https://github.com/cafferychen777/mLLMCelltype},
  version = {1.0.2},
  year = {2025}
}
```
