# mLLMCelltype <img src="man/figures/logo.png" align="right" height="139" alt="mLLMCelltype logo" />

[![GitHub stars](https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social)](https://github.com/cafferychen777/mLLMCelltype/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social)](https://github.com/cafferychen777/mLLMCelltype/network/members)
[![GitHub issues](https://img.shields.io/github/issues/cafferychen777/mLLMCelltype?style=social)](https://github.com/cafferychen777/mLLMCelltype/issues)
[![Paper](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.04.10.647852-blue)](https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1)

## Multi-LLM Consensus Architecture for Cell Type Annotation in scRNA-seq Data

mLLMCelltype is an R package that leverages various large language models (LLMs) for automated cell type annotation in single-cell RNA sequencing data. The package implements a **multi-LLM consensus architecture** where multiple LLMs collaborate through structured deliberation to provide more reliable annotations than any single model could achieve alone.

### Key Features

* **Multi-LLM Consensus Mechanism**: Harnesses collective intelligence from diverse LLMs to overcome single-model limitations and biases
* **Structured Deliberation Process**: For controversial clusters, LLMs engage in collaborative discussion across multiple rounds, evaluating evidence and refining annotations together
* **Uncertainty Quantification**: Explicitly quantifies annotation uncertainty through consensus proportion and Shannon entropy
* **No Reference Dataset Required**: Does not rely on pre-existing reference datasets, can annotate various tissues and species
* **Support for Multiple LLM Providers**:
  - OpenAI (GPT-4o, GPT-4.1, GPT-3.5-Turbo, O1 series)
  - Anthropic (Claude 3.7 Sonnet, Claude 3.5 Sonnet/Haiku/Opus)
  - Google (Gemini 2.5 Pro, Gemini 2.0, Gemini 1.5 series)
  - X.AI (Grok-3, Grok-3 Fast, Grok-3 Mini series)
  - DeepSeek (DeepSeek Chat, DeepSeek Reasoner)
  - Qwen (Qwen Max)
  - Zhipu (GLM-4 Plus, GLM-3 Turbo)
  - MiniMax (MiniMax Text)
  - Stepfun (Step-2, Step-1 series)
  - OpenRouter (access to Meta Llama, Mistral, Microsoft, Perplexity, Cohere, and more)
* **Seamless Integration with Seurat**: Can directly use Seurat's FindAllMarkers() output as input

### Quick Start

```r
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
  "claude-3-7-sonnet-20250219",
  "gpt-4o",
  "gemini-2.5-pro"
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

# Create consensus
consensus_results <- create_consensus(
  results = results,
  input = pbmc_markers,
  tissue_name = "human PBMC",
  model = "claude-3-7-sonnet-20250219",
  api_key = Sys.getenv("ANTHROPIC_API_KEY")
)

# Print consensus results summary
print_consensus_summary(consensus_results)
```

### Visualization

<img src="https://raw.githubusercontent.com/cafferychen777/mLLMCelltype/main/images/mLLMCelltype_visualization.png" alt="mLLMCelltype Visualization" width="900"/>

### Citation

If you use mLLMCelltype in your research, please cite our paper:

```bibtex
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

Yang, C., Zhang, X., & Chen, J. (2025). Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data. *bioRxiv*. https://doi.org/10.1101/2025.04.10.647852

### Learn More

Please check our [documentation](articles/01-introduction.html) to learn more about mLLMCelltype.
