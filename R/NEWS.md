# mLLMCelltype Changelog

## 1.2.5 (2025-06-02)

### Model Updates
* **Updated Gemini model support**: Added support for new Gemini models and removed discontinued ones
  - Added `gemini-2.0-flash-lite` to supported models list
  - Updated documentation to reflect Google's model migration recommendations
  - Removed references to discontinued Gemini 1.5 Pro 001 and Gemini 1.5 Flash 001 models

### Important Notes
* **Google Gemini Model Migration**: Google has discontinued several Gemini 1.5 models:
  - **Already discontinued**: Gemini 1.5 Pro 001, Gemini 1.5 Flash 001
  - **Will be discontinued on Sept 24, 2025**: Gemini 1.5 Pro 002, Gemini 1.5 Flash 002, Gemini 1.5 Flash-8B -001
  - **Recommended migration**: Use `gemini-2.0-flash` or `gemini-2.0-flash-lite` for better performance and continued support
  - The aliases `gemini-1.5-pro` and `gemini-1.5-flash` will continue to work until September 24, 2025

### Documentation Updates
* Updated all README files (English and international versions) with new Gemini model information
* Updated R documentation and vignettes to reflect model changes
* Added migration guidance in main documentation

## 1.2.4 (2025-05-25)

### Critical Bug Fixes
* **Fixed major `as.logical(from)` error**: Resolved critical error that occurred when processing large numbers of clusters (60+ clusters), which was caused by non-character data being passed to `strsplit()` functions
* **Enhanced error handling for API responses**: Added comprehensive `tryCatch()` blocks around all `strsplit()` operations in API processing functions
* **Improved response validation**: Added robust type checking for API responses to prevent function/closure types from being processed as character strings

### Improvements
* **Enhanced API processing robustness**: All API processing functions (`process_openrouter.R`, `process_anthropic.R`, `process_openai.R`, `process_deepseek.R`, `process_qwen.R`, `process_stepfun.R`, `process_minimax.R`, `process_zhipu.R`, `process_gemini.R`, `process_grok.R`) now include improved error handling
* **Better NULL value handling**: Improved `unlist()` operations to filter out NULL values and handle errors gracefully
* **Enhanced logging**: Added more detailed error logging for debugging API response issues
* **Improved consensus checking**: Enhanced `check_consensus.R` to handle edge cases with malformed responses

### Technical Details
* Fixed issue where large cluster datasets could cause type coercion errors in response parsing
* Added validation for function/closure types in API responses to prevent downstream errors
* Improved error messages to provide better diagnostics for API response issues

## 1.2.3 (2025-05-10)

### Bug Fixes
* Fixed error handling in consensus checking when API responses are NULL or invalid
* Improved error logging for OpenRouter API error responses
* Added robust NULL and type checking in check_consensus function

### Improvements
* Enhanced error diagnostics for OpenRouter API errors
* Added detailed logging of API error messages and response structures
* Improved robustness when handling unexpected API response formats

## 1.2.2 (2025-05-09)

### Bug Fixes
* Fixed the 'non-character argument' error that occurred when processing API responses
* Added robust type checking for API responses across all model providers
* Improved error handling for unexpected API response formats

### Improvements
* Added detailed error logging for API response issues
* Implemented consistent error handling patterns across all API processing functions
* Enhanced response validation to ensure proper structure before processing

## 1.2.1 (2025-05-01)

### Improvements
* Added support for OpenRouter API
* Added support for free models through OpenRouter
* Updated documentation with examples for using OpenRouter models

## 1.2.0 (2025-04-30)

### Features
* Added visualization functions for cell type annotation results
* Added support for uncertainty metrics visualization
* Implemented improved consensus building algorithm

## 1.1.5 (2025-04-27)

### Bug Fixes
* Fixed an issue with cluster index validation that caused errors when processing certain CSV input files
* Improved error handling for negative indices with clearer error messages

### Improvements
* Added example script for CSV-based annotation workflow (cat_heart_annotation.R)
* Enhanced input validation with more detailed diagnostics
* Updated documentation to clarify CSV input format requirements

## 1.1.4 (2025-04-24)

### Bug Fixes
* Fixed a critical issue with cluster index handling, now the package strictly accepts only 0-based indices (compatible with Seurat)
* Fixed negative index (-1) issues that could occur when processing CSV input files
* Added strict validation for input cluster indices to ensure they start from 0

### Improvements
* Removed automatic conversion logic from 1-based to 0-based indices in `prompt_templates.R`
* Added input validation in `consensus_annotation.R` to ensure cluster indices start from 0
* Updated code comments to clearly indicate that the package only accepts 0-based indices

## 1.1.3 (2025-04-15)

* Added support for X.AI's Grok models
* Updated the list of supported models, including Gemini 2.5 Pro
* Improved error handling and logging

## 1.1.2 (2025-03-30)

* Added support for Gemini 2.0 models
* Improved model response parsing
* Fixed cache management issues

## 1.1.1 (2025-03-15)

* Added support for Claude 3.7 and Claude 3.5 models
* Improved consensus building algorithm
* Fixed multiple minor bugs

## 1.1.0 (2025-03-01)

* Added interactive consensus annotation functionality
* Added multi-model discussion capability
* Improved cell type standardization

## 1.0.0 (2025-02-15)

* Initial release
* Support for cell type annotation using LLMs
* Support for models from OpenAI, Anthropic, and Google
