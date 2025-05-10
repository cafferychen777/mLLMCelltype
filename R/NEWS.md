# mLLMCelltype Changelog

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
