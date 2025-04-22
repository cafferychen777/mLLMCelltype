# Changelog

All notable changes to the Python implementation of mLLMCelltype will be documented in this file.

## [0.1.1] - 2025-04-22

### Added
- OpenRouter support for accessing multiple LLM providers through a single API
  - Added `process_openrouter` function in `providers/openrouter.py`
  - Updated provider mapping in `annotate.py` to include OpenRouter
  - Added OpenRouter API key handling in `utils.py`
  - Enhanced `get_provider` function to detect OpenRouter model format (provider/model-name)
  - Added support for using OpenRouter models in `interactive_consensus_annotation`
- New example script `openrouter_example.py` demonstrating OpenRouter integration
- Updated documentation with OpenRouter usage examples

### Changed
- Updated version number to 0.1.1
- Improved error handling in provider modules

## [0.1.0] - 2025-04-01

### Added
- Initial release of mLLMCelltype Python package
- Support for multiple LLM providers:
  - OpenAI (GPT-4o, etc.)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Alibaba (Qwen models)
  - DeepSeek
  - StepFun
  - Zhipu AI (GLM models)
  - MiniMax
  - X.AI (Grok models)
- Core functionality:
  - Single model annotation
  - Multi-model consensus annotation
  - Model comparison tools
  - AnnData/Scanpy integration
- Comprehensive documentation and examples
