# mLLMCelltype Changelog

## 1.3.6 (2026-02-07)

### Documentation Clarification
* Clarified current cluster ID contract in tutorials and examples: input cluster IDs are preserved as-is
* Updated mapping examples to use annotation names directly (instead of assuming sequential `0:(n-1)` IDs)
* Kept historical release notes unchanged; this entry documents the current behavior explicitly

## 1.3.5 (2025-10-12)

### Updated Models
* Updated Anthropic Claude model support to include latest releases:
  - Added **Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`) - Newest Sonnet model, same price as earlier Sonnets
  - Added **Claude Opus 4.1** (`claude-opus-4-1-20250805`) - Updated Opus model
  - Added Claude 4 series models (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`)
  - Added support for both date-versioned and alias formats (e.g., `claude-sonnet-4.5`, `claude-opus-4.1`)
* Updated all documentation, vignettes, and examples to recommend latest models
* All Sonnet models (4.5, 4, 3.5, 3.7) have identical pricing - Claude Sonnet 4.5 is now supported at identical pricing

### Documentation Updates
* Updated model recommendations in usage tutorials and vignettes
* Updated get_provider() documentation with comprehensive model lists
* Added clarification that dated model versions are identical across all platforms

### Notes
* OpenRouter now supports `anthropic/claude-sonnet-4.5` and `anthropic/claude-opus-4.1`
* All dated model versions (e.g., `20250929`) remain stable and do not change over time

## 1.3.4 (2025-09-09)

### Bug Fixes
* Fixed critical NA handling issue in base_api_processor.R that caused "missing value where TRUE/FALSE needed" errors
* Fixed similar NA handling issues in print_consensus_summary.R
* Improved robustness of API key validation to handle NA, NULL, and empty string cases correctly

### Improvements
* Enhanced error handling for invalid API keys
* Better support for OpenRouter API integration
* Improved stability when processing consensus annotations

## 1.3.2 (2025-09-02) - CRAN Release

### Official CRAN Release
* mLLMCelltype is now available on CRAN: `install.packages("mLLMCelltype")`
* CRAN page: https://CRAN.R-project.org/package=mLLMCelltype
* DOI: 10.32614/CRAN.package.mLLMCelltype

### Installation
* Standard R package installation from CRAN repository
* Package has passed CRAN checks for reliability and compatibility
* Tested across R versions and platforms (Windows, macOS, Linux)

### Features
* Automated cell type annotation using consensus from multiple LLMs
* Support for OpenAI, Anthropic, Google, and other providers through OpenRouter
* Integration with Seurat workflows
* Uncertainty quantification for annotations
* Caching system for API cost reduction
* Interactive consensus building with multi-model discussion

## 1.3.1 (2025-07-16)

### New Features: Enhanced Cluster Analysis Control

### Selective Cluster Analysis
* **New parameter `clusters_to_analyze`**: Allows users to specify exactly which clusters to analyze
  - No need to manually filter input data
  - Maintains original cluster numbering
  - Reduces API calls and costs by focusing on relevant clusters
  - Perfect for iterative refinement and subtyping workflows

### Cache Control for Re-analysis
* **New parameter `force_rerun`**: Forces fresh analysis of controversial clusters
  - Bypasses cache when you need new analysis with different context
  - Essential for subtype identification with tissue-specific context
  - Non-controversial clusters still benefit from cache performance
  - Combines perfectly with `clusters_to_analyze` for targeted workflows

### Use Cases
* **Iterative Subtyping**: Analyze broad cell types first, then focus on specific populations
* **Cost-Effective Re-analysis**: Only re-analyze controversial or specific clusters
* **Context-Specific Analysis**: Re-run with different tissue contexts without cache interference
* **Targeted Refinement**: Focus computational resources on clusters of interest

### Implementation
* Full backward compatibility maintained
* Comprehensive input validation and error handling
* Extensive testing with real API keys across multiple scenarios
* Updated documentation with detailed examples

## 1.2.9 (2025-06-24)

### Consensus check optimization

### Performance Optimization
* **Two-Stage Consensus Strategy**: Implemented optimized consensus checking that reduces redundant API calls through early consensus detection
  - **Stage 1 - Simple Consensus**: First performs fast local calculation based on normalized annotations
  - **Stage 2 - LLM Verification**: Only calls LLM for clusters that don't meet consensus thresholds
  - **Smart Resource Usage**: LLM only used for genuinely ambiguous cases

### Cost Reduction
* **API cost optimization**:
  - Clear consensus cases (majority of clusters) now processed without LLM calls
  - Reduces costs proportionally to API call reduction
  - Maintains same accuracy while being much more economical

### Implementation Details
* **New Functions**:
  - `normalize_annotation()`: Handles annotation variations (e.g., "T cells" vs "T lymphocytes")
  - `calculate_simple_consensus()`: Fast local consensus calculation with CP and entropy metrics
* **Improved Fallback**: If LLM fails, falls back to simple consensus results instead of default values
* **Enhanced Logging**: Clear indication of which strategy was used for each cluster

## 1.2.7 (2025-06-17)

### Major Enhancements: Code Deduplication and Enhanced Logging

### Code Architecture Refactoring
* **BaseAPIProcessor**: Introduced abstract base class for API processing with unified error handling and logging
  - Reduced code duplication across API processors
  - **Standardized interface**: All processors now inherit from `BaseAPIProcessor` with consistent `make_api_call()` and `extract_response_content()` methods
  - **Unified error handling**: Centralized exception handling and retry logic across all API providers

### Enhanced Discussion Logging
* **Fixed Discussion Format Issues**: Resolved fragmented discussion logs where multi-line model responses were split into multiple entries
  - **Improved readability**: Multi-line model predictions now properly consolidated into single coherent text blocks
  - **Better structure**: Clean Markdown format with proper headers and code block formatting
  - **Complete conversations**: Full model reasoning (grounds, warrant, backing, qualifier, rebuttal) preserved in readable format

### Complete API Request/Response Logging
* **New Feature**: Full API audit trail with complete request and response content
  - **Dual format logging**: Both JSON (machine-readable) and Markdown (human-readable) formats
  - **Complete transparency**: Every API call now logged with full prompt content and response
  - **Debugging support**: Detailed metadata including chunk information, timing, and response characteristics
  - **Session organization**: API logs organized in dedicated `api_logs/` subdirectories within each session

### Improved Log Organization
* **Enhanced directory structure**:
  ```
  logs/
  ├── mllm_YYYYMMDD_HHMMSS.log          # Main session log
  ├── YYYYMMDD_HHMMSS/                   # Session directory
  │   ├── cluster_X_discussion.md        # Discussion logs (fixed format)
  │   └── api_logs/                      # Complete API call logs (NEW)
  │       ├── provider_model_timestamp.json
  │       └── provider_model_timestamp.md
  ```

### Performance Improvements
* **Reduced codebase size**: Consolidated API processor code
* **Maintainability**: Single source of truth for API processing logic
* **Consistency**: Standardized error messages and logging across all providers

### Quality Assurance
* **Comprehensive testing**: All changes verified with real API calls using OpenRouter
* **Backward compatibility**: Existing functionality preserved while enhancing logging capabilities
* **Zero breaking changes**: All public APIs remain unchanged

## 1.2.6 (2025-06-14)

### Major Logging System Overhaul
* **Unified Logging System**: Replaced fragmented logging infrastructure with comprehensive unified logging
  - **Removed**: Old `DiscussionLogger` class and scattered `write_log` calls
  - **Added**: New `UnifiedLogger` R6 class with structured JSON output, multi-level logging, and performance monitoring
  - **New functions**: `configure_logger()`, `get_logger()`, `log_info()`, `log_warn()`, `log_error()`, `log_debug()`
  - **Features**: Session tracking, API call monitoring, cache operation logging, automatic log rotation

### Performance Improvements
* **Performance Monitoring**: Integrated performance tracking with session duration, API call counting, and error statistics
* **Log Rotation**: Automatic log file rotation with configurable size limits and file retention
* **Structured Logging**: JSON-formatted logs with context metadata for better analysis and debugging

### API Enhancements
* **Function Signature Updates**: Removed logger parameters from all consensus annotation functions
* **Code Cleanup**: Eliminated 20+ scattered logger parameter dependencies across the codebase
* **Cache Integration**: Enhanced cache operations with detailed logging and performance tracking

### Documentation Updates
* **Updated Vignettes**: Revised advanced features tutorial with new unified logging examples
* **Roxygen Documentation**: Updated all function documentation to reflect new logging system
* **NAMESPACE Updates**: Added exports for new logging functions and UnifiedLogger class

### Breaking Changes
* **Logger Parameters**: Removed `logger` parameters from `interactive_consensus_annotation()` and related functions
* **DiscussionLogger**: Deprecated `DiscussionLogger` class (still exported for backward compatibility but not recommended)

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
