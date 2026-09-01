# Package index

## Core Functions

Core functions for cell type annotation

- [`annotate_cell_types()`](https://cafferychen777.github.io/mLLMCelltype/reference/annotate_cell_types.md)
  : Cell Type Annotation with Multi-LLM Framework
- [`interactive_consensus_annotation()`](https://cafferychen777.github.io/mLLMCelltype/reference/interactive_consensus_annotation.md)
  : Interactive consensus building for cell type annotation
- [`compare_model_predictions()`](https://cafferychen777.github.io/mLLMCelltype/reference/compare_model_predictions.md)
  : Compare predictions from different models

## API and Providers

API processing functions for different LLM providers

- [`get_provider()`](https://cafferychen777.github.io/mLLMCelltype/reference/get_provider.md)
  : Determine provider from model name
- [`get_api_key()`](https://cafferychen777.github.io/mLLMCelltype/reference/get_api_key.md)
  : Get an API key for a model
- [`register_custom_model()`](https://cafferychen777.github.io/mLLMCelltype/reference/register_custom_model.md)
  : Register a custom model for a provider
- [`register_custom_provider()`](https://cafferychen777.github.io/mLLMCelltype/reference/register_custom_provider.md)
  : Register a custom LLM provider
- [`list_custom_providers()`](https://cafferychen777.github.io/mLLMCelltype/reference/list_custom_providers.md)
  : Get list of registered custom providers
- [`list_custom_models()`](https://cafferychen777.github.io/mLLMCelltype/reference/list_custom_models.md)
  : Get list of registered custom models

## Processor Classes

LLM processor classes for different providers

- [`BaseAPIProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/BaseAPIProcessor.md)
  : Base API Processor Class
- [`AnthropicProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/AnthropicProcessor.md)
  : Anthropic API Processor
- [`DeepSeekProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/DeepSeekProcessor.md)
  : DeepSeek API Processor
- [`GeminiProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/GeminiProcessor.md)
  : Gemini API Processor
- [`GrokProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/GrokProcessor.md)
  : Grok API Processor
- [`KimiProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/KimiProcessor.md)
  : Kimi API Processor
- [`MinimaxProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/MinimaxProcessor.md)
  : Minimax API Processor
- [`OpenAIProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/OpenAIProcessor.md)
  : OpenAI API Processor
- [`OpenRouterProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/OpenRouterProcessor.md)
  : OpenRouter API Processor
- [`QwenProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/QwenProcessor.md)
  : Qwen API Processor
- [`StepFunProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/StepFunProcessor.md)
  : StepFun API Processor
- [`ZhipuProcessor`](https://cafferychen777.github.io/mLLMCelltype/reference/ZhipuProcessor.md)
  : Zhipu API Processor

## Logging and Utilities

Logging system and utility functions

- [`UnifiedLogger`](https://cafferychen777.github.io/mLLMCelltype/reference/UnifiedLogger.md)
  : Unified Logger for mLLMCelltype Package
- [`configure_logger()`](https://cafferychen777.github.io/mLLMCelltype/reference/configure_logger.md)
  : Set global logger configuration
- [`get_logger()`](https://cafferychen777.github.io/mLLMCelltype/reference/get_logger.md)
  : Get the global logger instance
- [`log_debug()`](https://cafferychen777.github.io/mLLMCelltype/reference/logging_functions.md)
  [`log_info()`](https://cafferychen777.github.io/mLLMCelltype/reference/logging_functions.md)
  [`log_warn()`](https://cafferychen777.github.io/mLLMCelltype/reference/logging_functions.md)
  [`log_error()`](https://cafferychen777.github.io/mLLMCelltype/reference/logging_functions.md)
  : Convenience functions for logging
- [`CacheManager`](https://cafferychen777.github.io/mLLMCelltype/reference/CacheManager.md)
  : Cache Manager Class
- [`mllmcelltype_cache_dir()`](https://cafferychen777.github.io/mLLMCelltype/reference/mllmcelltype_cache_dir.md)
  : Get mLLMCelltype cache location
- [`mllmcelltype_clear_cache()`](https://cafferychen777.github.io/mLLMCelltype/reference/mllmcelltype_clear_cache.md)
  : Clear mLLMCelltype cache
- [`create_annotation_prompt()`](https://cafferychen777.github.io/mLLMCelltype/reference/create_annotation_prompt.md)
  : Create prompt for cell type annotation
- [`create_reasoning_annotation_prompt()`](https://cafferychen777.github.io/mLLMCelltype/reference/create_reasoning_annotation_prompt.md)
  : Create reasoning-aware prompt for cell type annotation
