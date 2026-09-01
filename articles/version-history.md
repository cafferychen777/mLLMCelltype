# Version History & Changelog

## Version History & Changelog

This document tracks the development history of mLLMCelltype, including
major releases, feature additions, bug fixes, and other significant
changes.

### Version 2.0.6 (2026-07-15)

#### Reliability

- Unified input, provider-response, consensus-metric, URL, and cache
  validation
- Centralized transient-failure retries at the provider boundary
- Made cluster alignment, marker selection, and cache behavior
  deterministic
- Added normalized provider token-usage and cost extraction

#### Maintenance

- Removed unused R dependencies
- Regenerated reference documentation from current source annotations
- Updated citation metadata for the Communications Biology publication

### Version 1.0.0 (2023-11-15)

#### Initial Release

- First public release of mLLMCelltype
- Core functionality for cell type annotation using LLMs
- Support for OpenAI (GPT-3.5, GPT-4) and Anthropic (Claude) models
- Basic consensus mechanism
- Integration with Seurat

### Version 1.1.0 (2024-01-20)

#### Features

- Added support for Google’s Gemini models
- Implemented structured deliberation process for controversial clusters
- Added uncertainty quantification with consensus proportion and Shannon
  entropy
- Improved caching system for API responses
- Added Python implementation

#### Bug Fixes

- Fixed issue with marker gene sorting
- Corrected handling of cluster indices
- Improved error messages for API failures

#### Documentation

- Added comprehensive README
- Created example notebooks
- Added function documentation

### Version 1.2.0 (2024-03-10)

#### Features

- Added support for DeepSeek, Qwen, and Zhipu models
- Implemented hierarchical annotation capability
- Added visualization functions for uncertainty metrics
- Improved handling of rate limits and API errors
- Added batch processing for large datasets

#### Bug Fixes

- Fixed consensus calculation for edge cases
- Corrected handling of empty API responses
- Improved error handling for network issues

#### Documentation

- Added benchmarking results
- Updated examples with new models
- Expanded troubleshooting guide

### Version 1.3.0 (2024-05-15)

#### Features

- Added support for Stepfun, MiniMax, and OpenRouter models
- Implemented custom prompt templates
- Added provider-specific parameter customization
- Improved performance with parallel processing
- Enhanced caching with persistent storage

#### Bug Fixes

- Fixed issue with discussion logs formatting
- Corrected handling of special characters in marker genes
- Improved robustness against API changes

#### Documentation

- Added case studies for different tissue types
- Created advanced usage guide
- Updated installation instructions for all dependencies

### Version 1.4.0 (2024-07-01)

#### Features

- Added support for Grok models from X.AI
- Updated Claude model support to include Claude 3.7 Sonnet
- Updated Gemini model support to include Gemini 2.5 Pro
- Improved consensus mechanism with weighted voting
- Enhanced visualization capabilities
- Added support for spatial transcriptomics data

#### Bug Fixes

- Fixed consensus result printing for controversial clusters
- Corrected model mapping for Claude 3.7 Sonnet
- Fixed undefined variable issue in consensus validation
- Improved Seurat object integration

#### Documentation

- Created comprehensive documentation website with pkgdown
- Added new vignettes for specific use cases
- Updated examples with latest models

### Version 1.4.1 (2024-07-15)

#### Bug Fixes

- Fixed issue with printing consensus results for controversial clusters
- Corrected model mapping in anthropic.py to properly use Claude 3.7
  Sonnet
- Fixed undefined has_names variable in consensus validation
- Improved test_pbmc3k.R and added to .gitignore to prevent API key
  leakage

#### Documentation

- Updated model lists in documentation
- Clarified API key setup instructions
- Improved error messages for common issues

### Breaking Changes

This section documents breaking changes that may require updates to your
code.

#### Version 1.2.0

- Changed the default value of `top_gene_count` from 5 to 10
- Modified the return structure of
  [`interactive_consensus_annotation()`](https://cafferychen777.github.io/mLLMCelltype/reference/interactive_consensus_annotation.md)
  to include additional metadata
- Renamed `uncertainty_score` to `shannon_entropy` for clarity

#### Version 1.3.0

- Changed the API for custom model registration
- Modified the caching system to use a different file structure
- Updated the required R version to 4.0.0 or higher

#### Version 1.4.0

- Changed the default models used in examples to Claude 3.7 and Gemini
  2.5
- Modified the return structure of
  [`annotate_cell_types()`](https://cafferychen777.github.io/mLLMCelltype/reference/annotate_cell_types.md)
  to include more metadata
- Updated the required package versions for several dependencies

### Deprecation Notices

The following features are deprecated and will be removed in future
versions:

- `simple_consensus()` function (since v1.2.0): Use
  [`interactive_consensus_annotation()`](https://cafferychen777.github.io/mLLMCelltype/reference/interactive_consensus_annotation.md)
  instead
- `basic_visualization()` function (since v1.3.0): Use the new
  visualization functions instead
- Support for older model versions will be gradually phased out as
  providers retire them

### Acknowledgments

We would like to thank all contributors who have helped improve
mLLMCelltype:

- Core developers: Chen Yang, \[List other core developers\]
- Contributors: \[List contributors\]
- Users who reported issues and suggested improvements

### How to Cite

If you use mLLMCelltype in your research, please cite:

    Yang, C., Zhang, X., & Chen, J. (2026). Large language model consensus substantially
    improves the cell type annotation accuracy for scRNA-seq data. Communications Biology.
    https://doi.org/10.1038/s42003-026-10420-8

### Feedback and Contributions

We welcome feedback and contributions to improve mLLMCelltype. Please
see our [Contributing
Guide](https://cafferychen777.github.io/mLLMCelltype/articles/contributing-guide.html)
for details on how to contribute.

### Next Steps

Now that you’ve reviewed the version history, you can:

- [Return to the
  introduction](https://cafferychen777.github.io/mLLMCelltype/articles/introduction.html)
  to get started with mLLMCelltype
- [Explore advanced
  features](https://cafferychen777.github.io/mLLMCelltype/articles/advanced-features.html)
  to learn about the latest capabilities
- [Check the
  FAQ](https://cafferychen777.github.io/mLLMCelltype/articles/faq.html)
  for answers to common questions
