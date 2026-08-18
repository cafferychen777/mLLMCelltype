# Changelog

All notable changes to the Python implementation of mLLMCelltype will be documented in this file.

## [2.0.8] - 2026-08-17

### Fixed
- Disabled DeepSeek V4 thinking mode for cell-type annotation requests so the
  response budget is reserved for final annotation text instead of being
  exhausted by internal reasoning.
- Increased the DeepSeek request timeout from 90 to 120 seconds for slower
  non-streaming responses.

## [2.0.7] - 2026-07-20

### Added
- Built-in Kimi provider targeting the Moonshot AI Open Platform
  (`https://api.moonshot.cn/v1/chat/completions`) via the OpenAI-compatible
  Chat Completions protocol. Models prefixed with `kimi-` or `moonshot-`
  resolve automatically (default model `kimi-k2.6`, API key from
  `MOONSHOT_API_KEY`), and Kimi k2 thinking mode is disabled for deterministic
  annotation output.
- The Kimi provider also accepts Kimi Code platform endpoints through
  `base_urls`: URLs ending in `/messages` (or the Kimi Code base
  `https://api.kimi.com/coding`) use the Anthropic Messages protocol, while
  `https://api.kimi.com/coding/v1` and other URLs use OpenAI-compatible Chat
  Completions.
- Added `return_reasoning` argument to `annotate_clusters()`. When set to
  `True`, the function returns a structured dictionary per cluster containing
  `cell_type`, `marker_genes`, and `gene_expression` fields instead of plain
  cell-type labels, enabling downstream inspection of the marker genes and
  expression rationale that support each annotation.

### Fixed
- Intermittent `Unknown` reasoning records in `annotate_clusters()` when
  `return_reasoning=True`. The provider pipeline now passes the raw response
  text through to reasoning JSON parsing instead of normalizing it into lines
  first, preserving commas and other JSON punctuation.
- Corrected labeled/positional parsing of model responses so common LLM output
  shapes no longer produce wrong or shifted annotations on 0-based
  (Seurat/Scanpy) clusters: numbered lists (`1.`, `2.`, …), preamble/header
  lines, annotation-internal colons, a mid-list `Unknown`, a stray non-cluster
  `Summary:`-style line, and out-of-order explicit labels are now all handled
  correctly.
- MiniMax transient in-body business errors (rate limits / timeouts reported as
  HTTP 200 with a non-zero `base_resp.status_code`) are now retried instead of
  silently dropping the model. A new `RetryableProviderError(RuntimeError)`
  stays within the consensus recovery net so a persistent transient error drops
  only that model, not the whole run.
- Anthropic responses that lead with a non-text (thinking) block no longer
  discard the answer; every text block is concatenated and non-text blocks are
  skipped.
- `is_error_response` no longer discards a valid multi-cluster response that
  merely flags a single uncertain cluster with an `Error:` line.
- Emit a warning when a provider response is truncated at `max_tokens`, so
  silently-dropped trailing clusters are diagnosable.

## [2.0.6] - 2026-07-15

### Added
- `interactive_consensus_annotation` now accepts an optional `prompt_template`
  argument, forwarded to the initial annotation phase (`create_prompt`). This
  exposes the existing per-prompt customization of `annotate_clusters` through
  the consensus entry point, enabling custom task framing / output contracts
  (e.g. functional-state annotation) without monkeypatching
  `DEFAULT_PROMPT_TEMPLATE`. Defaults to `None`, preserving current behavior.
- Optional normalized token-usage capture for Anthropic, Gemini, and
  OpenAI-compatible providers. A mutable `usage_sink` receives non-negative
  `prompt_tokens`, `completion_tokens`, and `total_tokens` values, plus native
  `cost` data when a provider returns it.
- Public usage extractors for callers that parse provider responses directly.

### Changed
- Centralized provider retry ownership and restricted retries to transient HTTP
  status codes and transport failures.
- Normalized prompt, model, API-key, base-URL, cluster-ID, and provider-response
  contracts across the public annotation and consensus entry points.
- Hardened cache envelopes, deterministic cache keys, and file-only cache
  maintenance operations.

### Fixed
- Honored custom Gemini base URLs and applied an explicit request timeout.
- Rejected malformed or internally inconsistent consensus metrics instead of
  accepting numerically plausible fragments.
- Prevented ambiguous cluster labels and model identities from being silently
  reassigned during consensus processing.

### Documentation
- Updated paper citation metadata to the Communications Biology publication DOI.

## [2.0.5] - 2026-05-11

### Fixed
- Unified discussion report handling in examples with the package formatter.
- Updated examples to use current model identifiers and correct local import paths.

### Documentation
- Updated the package badge for the 2.0.5 release.

## [2.0.4] - 2026-05-10

### Fixed
- Made the complex consensus test an explicit integration test so default pytest runs stay offline and deterministic.
- Aligned Python documentation with the package metadata and current optional dependencies.
- Clarified that providers using OpenAI-compatible HTTP endpoints do not require nonexistent provider extras.

## [1.2.5] - 2025-10-12

### Updated
- Updated Anthropic Claude model list to include latest models:
  - Added **Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`) - Newest Sonnet model
  - Added **Claude Opus 4.1** (`claude-opus-4-1-20250805`) - Updated model
  - Added Claude 4 series models with date versions
  - All Sonnet models (4.5, 4, 3.5, 3.7) have the same pricing - recommend using latest version
- Updated all documentation and examples to use latest model recommendations
- Updated model migration suggestions for deprecated Claude models

### Notes
- Claude Sonnet 4.5 is recommended for general use at same price as earlier Sonnet versions
- All dated model versions (e.g., 20250929) are identical across platforms and do not change

## [1.2.4] - 2025-06-24

### Added
- **Consensus Check Optimization**: Implemented two-stage consensus checking strategy
  - First performs simple consensus calculation based on normalized annotations
  - Only calls LLM for clusters that don't meet consensus thresholds
  - Reduced API calls through consensus optimization
  - Maintains same accuracy while reducing costs

### Changed
- Modified `check_consensus()` function to prioritize simple consensus checks
- Added detailed logging to track when simple consensus is sufficient vs when LLM is needed

## [1.2.3] - 2025-06-03

### Fixed
- Fixed `UnboundLocalError: local variable 'consensus_response' referenced before assignment` in `process_controversial_clusters` function
- Added proper initialization and null checks for `consensus_response` variable to prevent crashes when consensus is not reached within maximum discussion rounds
- Improved error handling in controversial cluster resolution process

## [1.2.2] - 2025-06-02

### Updated
- Updated Gemini model list to include new models and remove discontinued ones:
  - Added support for `gemini-3-pro` and `gemini-2.0-flash-lite`
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
