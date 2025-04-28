# mLLMCelltype <img src="man/figures/logo.png" align="right" height="139" alt="mLLMCelltype logo" />

<div align="center">
  <a href="https://twitter.com/intent/tweet?text=Check%20out%20mLLMCelltype%3A%20A%20multi-LLM%20consensus%20framework%20for%20cell%20type%20annotation%20in%20scRNA-seq%20data%21&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype"><img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fcafferychen777%2FmLLMCelltype" alt="Tweet"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/stargazers"><img src="https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social" alt="Stars"></a>
  <a href="https://github.com/cafferychen777/mLLMCelltype/network/members"><img src="https://img.shields.io/github/forks/cafferychen777/mLLMCelltype?style=social" alt="Forks"></a>
  <a href="https://discord.gg/pb2aZdG4"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</div>

<div align="center">
  <img src="https://img.shields.io/github/license/cafferychen777/mLLMCelltype" alt="License">
  <img src="https://img.shields.io/github/last-commit/cafferychen777/mLLMCelltype" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/cafferychen777/mLLMCelltype" alt="Issues">
  <img src="https://img.shields.io/github/v/release/cafferychen777/mLLMCelltype" alt="Release">
  <a href="https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1"><img src="https://img.shields.io/badge/bioRxiv-2025.04.10.647852-blue" alt="bioRxiv"></a>
</div>

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
  consensus_check_model = "claude-3-7-sonnet-20250219"
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
