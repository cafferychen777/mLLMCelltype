# Changelog

All notable changes to the Python implementation of mLLMCelltype will be documented in this file.

## [1.2.3] - 2025-06-03

### Fixed
- Fixed `UnboundLocalError: local variable 'consensus_response' referenced before assignment` in `process_controversial_clusters` function
- Added proper initialization and null checks for `consensus_response` variable to prevent crashes when consensus is not reached within maximum discussion rounds
- Improved error handling in controversial cluster resolution process

## [1.2.2] - 2025-06-02

### Updated
- Updated Gemini model list to include new models and remove discontinued ones:
  - Added support for `gemini-2.5-pro` and `gemini-2.0-flash-lite`
  - Removed discontinued `gemini-2.0-flash-001` model
  - Updated documentation to reflect Google's model migration recommendations

### Notes
- Google has discontinued Gemini 1.5 Pro 001 and Gemini 1.5 Flash 001 models
- Gemini 1.5 Pro 002, Gemini 1.5 Flash 002, and Gemini 1.5 Flash-8B -001 will be discontinued on September 24, 2025
- Users are recommended to migrate to `gemini-2.0-flash` or `gemini-2.0-flash-lite` for better performance

## [1.2.1] - 2025-04-29

### Added
- Added support for Alibaba's Qwen3-72B model

## [1.2.0] - 2025-04-23

### Added
- Added support for X.AI's Grok models:
  - grok-3, grok-3-latest
  - grok-3-fast, grok-3-fast-latest
  - grok-3-mini, grok-3-mini-latest
  - grok-3-mini-fast, grok-3-mini-fast-latest
- Added support for Google's Gemini 2.5 Pro model
- Enhanced OpenRouter integration with improved error handling

### Fixed
- Fixed Claude 3.7 Sonnet model (claude-3-7-sonnet-20250219) being incorrectly mapped to Claude 3.5 Sonnet model (claude-3-5-sonnet-20240620)
- Updated linting configuration to use ruff instead of black for consistent code formatting

## [1.1.0] - 2025-04-21

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
- Updated version number to 1.1.0
- Improved error handling in provider modules

## [1.0.0] - 2025-03-15

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
