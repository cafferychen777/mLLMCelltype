# Package index

## Core Functions

Core functions for cell type annotation

- [`annotate_cell_types()`](https://cafferyang.com/mLLMCelltype/reference/annotate_cell_types.md)
  : Cell Type Annotation with Multi-LLM Framework
- [`interactive_consensus_annotation()`](https://cafferyang.com/mLLMCelltype/reference/interactive_consensus_annotation.md)
  : Interactive consensus building for cell type annotation
- [`compare_model_predictions()`](https://cafferyang.com/mLLMCelltype/reference/compare_model_predictions.md)
  : Compare predictions from different models

## API and Providers

API processing functions for different LLM providers

- [`get_provider()`](https://cafferyang.com/mLLMCelltype/reference/get_provider.md)
  : Determine provider from model name
- [`get_api_key()`](https://cafferyang.com/mLLMCelltype/reference/get_api_key.md)
  : Utility functions for API key management
- [`register_custom_model()`](https://cafferyang.com/mLLMCelltype/reference/register_custom_model.md)
  : Register a custom model for a provider
- [`register_custom_provider()`](https://cafferyang.com/mLLMCelltype/reference/register_custom_provider.md)
  : Register a custom LLM provider
- [`list_custom_providers()`](https://cafferyang.com/mLLMCelltype/reference/list_custom_providers.md)
  : Get list of registered custom providers
- [`list_custom_models()`](https://cafferyang.com/mLLMCelltype/reference/list_custom_models.md)
  : Get list of registered custom models

## Processor Classes

LLM processor classes for different providers

- [`BaseAPIProcessor`](https://cafferyang.com/mLLMCelltype/reference/BaseAPIProcessor.md)
  : Base API Processor Class
- [`AnthropicProcessor`](https://cafferyang.com/mLLMCelltype/reference/AnthropicProcessor.md)
  : Anthropic API Processor
- [`DeepSeekProcessor`](https://cafferyang.com/mLLMCelltype/reference/DeepSeekProcessor.md)
  : DeepSeek API Processor
- [`GeminiProcessor`](https://cafferyang.com/mLLMCelltype/reference/GeminiProcessor.md)
  : Gemini API Processor
- [`GrokProcessor`](https://cafferyang.com/mLLMCelltype/reference/GrokProcessor.md)
  : Grok API Processor
- [`MinimaxProcessor`](https://cafferyang.com/mLLMCelltype/reference/MinimaxProcessor.md)
  : Minimax API Processor
- [`OpenAIProcessor`](https://cafferyang.com/mLLMCelltype/reference/OpenAIProcessor.md)
  : OpenAI API Processor
- [`OpenRouterProcessor`](https://cafferyang.com/mLLMCelltype/reference/OpenRouterProcessor.md)
  : OpenRouter API Processor
- [`QwenProcessor`](https://cafferyang.com/mLLMCelltype/reference/QwenProcessor.md)
  : Qwen API Processor
- [`StepFunProcessor`](https://cafferyang.com/mLLMCelltype/reference/StepFunProcessor.md)
  : StepFun API Processor
- [`ZhipuProcessor`](https://cafferyang.com/mLLMCelltype/reference/ZhipuProcessor.md)
  : Zhipu API Processor

## Logging and Utilities

Logging system and utility functions

- [`UnifiedLogger`](https://cafferyang.com/mLLMCelltype/reference/UnifiedLogger.md)
  : Unified Logger for mLLMCelltype Package
- [`configure_logger()`](https://cafferyang.com/mLLMCelltype/reference/configure_logger.md)
  : Set global logger configuration
- [`get_logger()`](https://cafferyang.com/mLLMCelltype/reference/get_logger.md)
  : Get the global logger instance
- [`log_debug()`](https://cafferyang.com/mLLMCelltype/reference/logging_functions.md)
  [`log_info()`](https://cafferyang.com/mLLMCelltype/reference/logging_functions.md)
  [`log_warn()`](https://cafferyang.com/mLLMCelltype/reference/logging_functions.md)
  [`log_error()`](https://cafferyang.com/mLLMCelltype/reference/logging_functions.md)
  : Convenience functions for logging
- [`CacheManager`](https://cafferyang.com/mLLMCelltype/reference/CacheManager.md)
  : Cache Manager Class
- [`mllmcelltype_cache_dir()`](https://cafferyang.com/mLLMCelltype/reference/mllmcelltype_cache_dir.md)
  : Get mLLMCelltype cache location
- [`mllmcelltype_clear_cache()`](https://cafferyang.com/mLLMCelltype/reference/mllmcelltype_clear_cache.md)
  : Clear mLLMCelltype cache
- [`create_annotation_prompt()`](https://cafferyang.com/mLLMCelltype/reference/create_annotation_prompt.md)
  : Create prompt for cell type annotation
