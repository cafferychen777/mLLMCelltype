# mLLMCelltype: Overview and Quick Reference

``` r

library(mLLMCelltype)
```

## Overview

mLLMCelltype is an R package that leverages various large language
models (LLMs) for automated cell type annotation in single-cell RNA
sequencing data. It implements a consensus-based approach where multiple
LLMs collaborate to provide more reliable annotations. For more details,
please refer to our paper (<doi:10.1101/2025.04.10.647852>).

## Installation

You can install the released version of mLLMCelltype from CRAN:

``` r

install.packages("mLLMCelltype")
```

Or install the development version from GitHub:

``` r

# install.packages("devtools")
devtools::install_github("cafferychen777/mLLMCelltype")
```

## Setting Up API Keys

Before using mLLMCelltype, you need to set up API keys for the LLMs you
want to use. The package supports multiple LLM providers including
OpenAI, Anthropic, Google (Gemini), and X.AI (Grok).

``` r

# Set API keys (recommended to use .Renviron or .env file)
# OpenAI API key
Sys.setenv(OPENAI_API_KEY = "your-openai-api-key")

# Anthropic API key
Sys.setenv(ANTHROPIC_API_KEY = "your-anthropic-api-key")

# Google API key
Sys.setenv(GOOGLE_API_KEY = "your-google-api-key")

# X.AI API key
Sys.setenv(XAI_API_KEY = "your-xai-api-key")

# Alternatively, use .env file
# dotenv::load_dot_env()
```

## Basic Usage

### Annotating Cell Types with Seurat Object

The main function of mLLMCelltype is to annotate cell types in a Seurat
object:

``` r

# Load required packages
library(mLLMCelltype)
library(Seurat)
library(dplyr)

# Load your preprocessed Seurat object
# pbmc <- readRDS("your_seurat_object.rds")

# Find marker genes for each cluster
# pbmc_markers <- FindAllMarkers(pbmc,
#                           only.pos = TRUE,
#                           min.pct = 0.25,
#                           logfc.threshold = 0.25)

# Cache will be automatically managed using system cache directory

# Run LLMCelltype annotation with multiple LLM models
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",  # provide tissue context
  models = c(
    "claude-sonnet-4-6",  # Anthropic
    "gpt-5.5",                   # OpenAI
    "gemini-3.1-pro-preview"            # Google
  ),
  api_keys = list(
    anthropic = Sys.getenv("ANTHROPIC_API_KEY"),
    openai = Sys.getenv("OPENAI_API_KEY"),
    gemini = Sys.getenv("GOOGLE_API_KEY")
  ),
  top_gene_count = 10,
  controversy_threshold = 1.0,
  entropy_threshold = 1.0,
  consensus_check_model = "claude-sonnet-4-6",  # Use a reliable model for consensus checking
  cache_dir = NULL  # Uses default system cache directory
)

# Add annotations to Seurat object
# Get cell type annotations from consensus_results$final_annotations
cluster_to_celltype_map <- consensus_results$final_annotations

# Create new cell type identifier column
cell_types <- as.character(Idents(pbmc))
for (cluster_id in names(cluster_to_celltype_map)) {
  cell_types[cell_types == cluster_id] <- cluster_to_celltype_map[[cluster_id]]
}

# Add cell types to Seurat object
pbmc$mLLM_cell_type <- cell_types
```

### Visualizing Results

``` r

# Plot UMAP with cell type annotations
library(Seurat)
library(ggplot2)
library(cowplot)

# Basic visualization
p1 <- DimPlot(pbmc, group.by = "mLLM_cell_type", label = TRUE) +
  ggtitle("mLLMCelltype Consensus Annotations")

# Visualize consensus proportion (confidence metric)
pbmc$consensus_proportion <- 0
for (cluster_id in names(consensus_results$initial_results$consensus_results)) {
  cluster_cells <- which(Idents(pbmc) == cluster_id)
  pbmc$consensus_proportion[cluster_cells] <-
    consensus_results$initial_results$consensus_results[[cluster_id]]$consensus_proportion
}

p2 <- FeaturePlot(pbmc, features = "consensus_proportion",
                 min.cutoff = 0, max.cutoff = 1) +
  scale_color_gradientn(colors = c("blue", "green", "red")) +
  ggtitle("Consensus Proportion") +
  theme(legend.position = "right")
```

## Supported Models

mLLMCelltype supports various LLM models:

- **OpenAI**: gpt-5.5, gpt-5.4, gpt-5.4-mini
- **Anthropic**: claude-opus-4-7, claude-sonnet-4-6,
  claude-haiku-4-5-20251001
- **DeepSeek**: deepseek-v4-flash, deepseek-v4-pro
- **Google**: gemini-3.1-pro-preview, gemini-3-flash-preview,
  gemini-3.1-flash-lite
- **Qwen**: qwen3.6-plus, qwen3.6-flash, qwen3.6-max-preview
- **Stepfun**: step-3.5-flash, step-3.5-flash-2603, step-3
- **Zhipu/Z.AI**: glm-5.1, glm-5, glm-5-turbo
- **MiniMax**: MiniMax-M2.7, MiniMax-M2.7-highspeed, MiniMax-M2.5
- **Grok**: grok-4.3, grok-4.3-latest, grok-latest
- **OpenRouter**: Access to models from multiple providers through a
  single API

## Advanced Usage

### Using a Single LLM Model

If you only want to use a single LLM model instead of the consensus
approach:

``` r

# Run annotation with a single model
single_model_results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "claude-sonnet-4-6",
  api_key = Sys.getenv("ANTHROPIC_API_KEY"),
  top_gene_count = 10
)

# Inspect the provider response lines. Parse or review this text before
# mapping annotations back onto a Seurat object.
cat(single_model_results)
```

### Customizing Consensus Parameters

``` r

# Customize consensus parameters
custom_consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  models = c("claude-sonnet-4-6", "gpt-5.5", "gemini-3.1-pro-preview"),
  api_keys = list(
    anthropic = Sys.getenv("ANTHROPIC_API_KEY"),
    openai = Sys.getenv("OPENAI_API_KEY"),
    gemini = Sys.getenv("GOOGLE_API_KEY")
  ),
  top_gene_count = 15,                # Number of top marker genes to use
  controversy_threshold = 0.7,        # Threshold for controversy detection
  entropy_threshold = 0.7,            # Entropy threshold for controversy detection
  max_discussion_rounds = 2,          # Maximum rounds of discussion
  consensus_check_model = "claude-sonnet-4-6",  # Use a reliable model for consensus checking
  cache_dir = NULL  # Uses default system cache directory
)
```

### Using Custom Providers

You can register custom LLM providers:

``` r

# Register a custom provider
register_custom_provider(
  provider_name = "my_custom_provider",
  process_fn = my_process_function,
  description = "My custom LLM provider"
)

# Register a model served by that provider
register_custom_model(
  model_name = "my-model-1",
  provider_name = "my_custom_provider",
  model_config = list()
)

# Use the custom model
results <- annotate_cell_types(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "my-model-1",
  api_key = Sys.getenv("MY_CUSTOM_PROVIDER_API_KEY")
)
```

## Caching Results

mLLMCelltype includes a caching system to save API costs:

``` r

# Enable caching
Sys.setenv(MLLM_CACHE_ENABLED = "TRUE")

# Set cache directory
Sys.setenv(MLLM_CACHE_DIR = "path/to/cache")

# Clear cache if needed
mllmcelltype_clear_cache()
```

## Conclusion

mLLMCelltype is a framework for automated cell type annotation using
large language models. By combining predictions from multiple models, it
aims to produce more reliable annotations than single-model approaches.

For bug reports or feature requests, please visit:
<https://github.com/cafferychen777/mLLMCelltype/issues>.
