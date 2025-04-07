# mLLMCelltype

mLLMCelltype is an iterative multi-LLM consensus framework for cell type annotation in single-cell RNA sequencing data. By leveraging the complementary strengths of multiple large language models (GPT-4o, Claude-3.5, Gemini, Qwen, etc.), this framework significantly improves annotation accuracy while providing transparent uncertainty quantification.

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
from mllmcelltype import mLLMCelltype
import scanpy as sc

# Load your preprocessed data
adata = sc.read("your_data.h5ad")

# Initialize the framework with your API keys
annotator = mLLMCelltype(
    openai_api_key="your_openai_key",
    anthropic_api_key="your_anthropic_key",
    google_api_key="your_google_key"
)

# Run annotation with tissue context
results = annotator.annotate(
    adata=adata,
    tissue="lung",
    max_rounds=3,
    consensus_threshold=0.67
)

# Add annotations back to AnnData object
adata.obs['mLLM_celltype'] = results['annotations']
adata.obs['consensus_proportion'] = results['consensus_proportion']
adata.obs['entropy'] = results['entropy']
```

### R

```r
library(mLLMCelltype)
library(Seurat)

# Load your preprocessed Seurat object
seurat_obj <- readRDS("your_data.rds")

# Initialize the framework with your API keys
annotator <- init_mLLMCelltype(
    openai_api_key = "your_openai_key",
    anthropic_api_key = "your_anthropic_key",
    google_api_key = "your_google_key"
)

# Run annotation with tissue context
results <- annotate_seurat(
    seurat_obj = seurat_obj,
    tissue = "lung",
    max_rounds = 3,
    consensus_threshold = 0.67
)

# Add annotations back to Seurat object
seurat_obj$mLLM_celltype <- results$annotations
seurat_obj$consensus_proportion <- results$consensus_proportion
seurat_obj$entropy <- results$entropy
```

## License

MIT
