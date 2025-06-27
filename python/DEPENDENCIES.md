# mLLMCelltype Python Dependencies

## Core Dependencies (Required)

These dependencies are automatically installed when you install mLLMCelltype:

- **pandas** ≥ 1.0.0 - Data manipulation and analysis
- **numpy** ≥ 1.19.0 - Numerical computing
- **requests** ≥ 2.25.0 - HTTP library for API calls
- **python-dotenv** ≥ 0.19.0 - Environment variable management
- **jsonschema** ≥ 4.0.0 - JSON schema validation

## Optional Dependencies

### For specific LLM providers

Install with: `pip install mllmcelltype[provider_name]`

- **OpenAI**: `pip install mllmcelltype[openai]`
  - openai ≥ 1.0.0

- **Anthropic**: `pip install mllmcelltype[anthropic]`
  - anthropic ≥ 0.5.0

- **Google Gemini**: `pip install mllmcelltype[gemini]`
  - google-genai ≥ 1.0.0

- **X.AI Grok**: `pip install mllmcelltype[grok]`
  - x-ai ≥ 0.1.0

### For visualization

Install with: `pip install mllmcelltype[visualization]`

- matplotlib ≥ 3.3.0
- seaborn ≥ 0.11.0

**Note**: These visualization libraries are not used in the core mLLMCelltype package. They are provided as optional dependencies for users who want to create their own visualizations of the annotation results. If you're using mLLMCelltype with scanpy, these libraries are already included as scanpy's required dependencies.

### For running examples

Install with: `pip install mllmcelltype[examples]`

- matplotlib ≥ 3.3.0
- scanpy ≥ 1.8.0
- python-dotenv ≥ 0.19.0

### Install all optional dependencies

Install with: `pip install mllmcelltype[all]`

This includes:
- All LLM provider libraries (OpenAI, Anthropic, Google)
- Visualization libraries (matplotlib, seaborn)
- Environment management (python-dotenv)

### For development

Install with: `pip install mllmcelltype[dev]`

- pytest ≥ 6.0.0 - Testing framework
- pytest-cov ≥ 2.12.0 - Test coverage
- pre-commit ≥ 2.16.0 - Git pre-commit hooks
- ruff ≥ 0.11.0 - Python linter and formatter

## Python Version Requirement

- Python ≥ 3.9 (supports 3.9, 3.10, 3.11)

## Installation Examples

```bash
# Basic installation (core dependencies only)
pip install mllmcelltype

# Install with OpenAI support
pip install mllmcelltype[openai]

# Install with multiple providers
pip install mllmcelltype[openai,anthropic,gemini]

# Install everything (recommended for first-time users)
pip install mllmcelltype[all]

# Install for development
pip install -e .[dev]
```

## Notes

1. **API Keys**: While the provider libraries are optional, you'll need API keys for the LLM services you want to use.

2. **Minimal Installation**: If you're only using models through OpenRouter or providers that use standard HTTP requests, you may not need to install provider-specific libraries.

3. **Version Compatibility**: The package is tested with Python 3.9, 3.10, and 3.11. We recommend using the latest stable Python version for best performance.

4. **Memory Requirements**: Running multiple LLMs for consensus annotation may require significant memory, especially for large datasets. We recommend at least 8GB RAM for typical use cases.