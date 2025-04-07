# mLLMCelltype

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
    models=["gpt-4o", "claude-3-5-sonnet-latest", "gemini-1.5-pro", "qwen-max-2025-01-25"],
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

# Visualize results
sc.pl.umap(adata, color='consensus_cell_type', legend_loc='on data')
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
    "claude-3-5-sonnet-latest",  # Anthropic
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

# Visualize results
DimPlot(pbmc, reduction = "umap", group.by = "cell_type", label = TRUE) +
  labs(title = "mLLMCelltype Consensus Annotations")
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
  version = {1.0.0},
  year = {2025}
}
```
