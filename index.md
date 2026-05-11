# mLLMCelltype

![mLLMCelltype logo](reference/figures/logo.png)

**Multi-LLM Consensus Framework for Cell Type Annotation in scRNA-seq
Data**

[![Stars](https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social)](https://github.com/cafferychen777/mLLMCelltype/stargazers)
[![Forks](https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social)](https://github.com/cafferychen777/mLLMCelltype/network/members)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da?logo=discord&logoColor=white)](https://discord.gg/pb2aZdG4)

![License](https://img.shields.io/github/license/cafferychen777/mLLMCelltype)![Last
Commit](https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype)![Issues](https://img.shields.io/github/issues/cafferychen777/mLLMCelltype)![Release](https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype)[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue)](https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1)

## Multi-LLM Consensus Architecture for Cell Type Annotation in scRNA-seq Data

mLLMCelltype is an R package that leverages various large language
models (LLMs) for automated cell type annotation in single-cell RNA
sequencing data. The package implements a **multi-LLM consensus
architecture** where multiple LLMs collaborate through structured
deliberation that aims to improve annotation reliability by combining
multiple model predictions.

### Key Features

- **Multi-LLM Consensus Mechanism**: Combines predictions from multiple
  LLMs to reduce individual model biases
- **Structured Deliberation Process**: For controversial clusters, LLMs
  engage in collaborative discussion across multiple rounds, evaluating
  evidence and refining annotations together
- **Uncertainty Quantification**: Explicitly quantifies annotation
  uncertainty through consensus proportion and Shannon entropy
- **No Reference Dataset Required**: Does not rely on pre-existing
  reference datasets, can annotate various tissues and species
- **Support for Multiple LLM Providers**:
  - OpenAI (GPT-5.5, GPT-5.4, GPT-5.4-mini)
  - Anthropic (Claude Opus 4.7, Claude Sonnet 4.6, Claude Haiku 4.5)
  - Google (Gemini 3.1 Pro Preview, Gemini 3 Flash Preview, Gemini 3.1
    Flash-Lite)
  - X.AI (Grok 4.3)
  - DeepSeek (DeepSeek V4 Flash, DeepSeek V4 Pro)
  - Qwen (Qwen3.6 Plus, Qwen3.6 Flash, Qwen3.6 Max Preview)
  - Zhipu/Z.AI (GLM-5.1, GLM-5, GLM-5-Turbo)
  - MiniMax (MiniMax M2.7, MiniMax M2.7-highspeed)
  - Stepfun (Step 3.5 Flash, Step 3)
  - OpenRouter (access to Meta Llama, Mistral, Microsoft, Perplexity,
    Cohere, and more)
- **Seurat Integration**: Can directly use Seurat’s FindAllMarkers()
  output as input

### Quick Start

``` r

# Install the package
devtools::install_github("cafferychen777/mLLMCelltype", subdir = "R")

# Load the package
library(mLLMCelltype)

# Set API keys
Sys.setenv(ANTHROPIC_API_KEY = "your-anthropic-api-key")
Sys.setenv(OPENAI_API_KEY = "your-openai-api-key")
Sys.setenv(GEMINI_API_KEY = "your-gemini-api-key")

# Use multiple models for annotation
models <- c(
  "claude-sonnet-4-6",
  "gpt-5.5",
  "gemini-3.1-pro-preview"
)

# Run multi-model annotation
results <- list()
for (model in models) {
  provider <- get_provider(model)
  api_key <- switch(provider,
                   "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
                   "openai" = Sys.getenv("OPENAI_API_KEY"),
                   "gemini" = Sys.getenv("GEMINI_API_KEY"))

  results[[model]] <- annotate_cell_types(
    input = pbmc_markers,
    tissue_name = "human PBMC",
    model = model,
    api_key = api_key
  )
}

# Create consensus using interactive consensus annotation
api_keys <- list(
  anthropic = Sys.getenv("ANTHROPIC_API_KEY"),
  openai = Sys.getenv("OPENAI_API_KEY"),
  gemini = Sys.getenv("GEMINI_API_KEY")
)

consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  models = models,  # Use the models defined above
  api_keys = api_keys,
  controversy_threshold = 0.7,
  entropy_threshold = 1.0,
  max_discussion_rounds = 3,
  consensus_check_model = "claude-sonnet-4-6"
)
```

### Visualization

![mLLMCelltype
Visualization](https://raw.githubusercontent.com/cafferychen777/mLLMCelltype/main/images/mLLMCelltype_visualization.png)

### Citation

If you use mLLMCelltype in your research, please cite our paper:

``` bibtex
@article{Yang2025.04.10.647852,
  author = {Chen Yang and Xianyang Zhang and Jun Chen},
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

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus
Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq
Data. *bioRxiv*. <https://doi.org/10.1101/2025.04.10.647852>

### Learn More

Please check our
[documentation](https://cafferyang.com/mLLMCelltype/articles/01-introduction.md)
to learn more about mLLMCelltype.
