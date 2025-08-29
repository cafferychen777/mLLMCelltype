# LangExtract Configuration Guide

This guide covers the complete configuration system for LangExtract integration in mLLMCelltype.

## Overview

The LangExtract configuration system provides:
- **Environment variable support** for all configuration options
- **Flexible configuration hierarchy** with clear priority rules
- **Automatic validation** and error handling
- **Integration with existing API key management**
- **Both R and Python support** with consistent interfaces

## Configuration Hierarchy

Configuration values are resolved in the following priority order (highest to lowest):

1. **Function parameters** - Direct arguments passed to functions
2. **Environment variables** - Set via `.env` file or system environment
3. **Default values** - Built-in sensible defaults

## Quick Start

### 1. Set up Environment Variables

Copy the example environment file:

```bash
cp langextract.env.example .env
```

Edit `.env` and configure your settings:

```bash
# Enable LangExtract
LANGEXTRACT_ENABLED=true

# Choose your model
LANGEXTRACT_MODEL=gemini-2.5-flash

# Set complexity threshold
LANGEXTRACT_COMPLEXITY_THRESHOLD=0.7

# Add your API key
GEMINI_API_KEY=your_api_key_here
```

### 2. R Usage Example

```r
library(mLLMCelltype)

# Load configuration from environment
config <- load_langextract_config()

# Print configuration
print_langextract_config(config)

# Check configuration health
health <- check_langextract_config_health(config)
print(health)

# Use with function parameters (highest priority)
config_with_override <- load_langextract_config(
  config_override = list(
    model = "gpt-4o-mini",
    complexity_threshold = 0.8
  )
)
```

### 3. Python Usage Example

```python
from mllmcelltype.langextract_config import (
    load_langextract_config,
    print_langextract_config,
    check_langextract_config_health
)

# Load configuration from environment
config = load_langextract_config()

# Print configuration
print_langextract_config(config)

# Check configuration health
health = check_langextract_config_health(config)
print(health)

# Use with overrides
config_with_override = load_langextract_config({
    "model": "gpt-4o-mini",
    "complexity_threshold": 0.8
})
```

## Configuration Options

### Core Settings

| Setting | Environment Variable | Type | Default | Description |
|---------|---------------------|------|---------|-------------|
| `enabled` | `LANGEXTRACT_ENABLED` | boolean | `true` | Enable/disable LangExtract |
| `model` | `LANGEXTRACT_MODEL` | string | `"gemini-2.5-flash"` | Model to use |
| `complexity_threshold` | `LANGEXTRACT_COMPLEXITY_THRESHOLD` | float | `0.7` | Complexity threshold (0.0-1.0) |
| `fallback_enabled` | `LANGEXTRACT_FALLBACK_ENABLED` | boolean | `true` | Enable fallback mode |

### Performance Settings

| Setting | Environment Variable | Type | Default | Description |
|---------|---------------------|------|---------|-------------|
| `cache_enabled` | `LANGEXTRACT_CACHE_ENABLED` | boolean | `true` | Enable result caching |
| `api_timeout` | `LANGEXTRACT_API_TIMEOUT` | integer | `30` | API timeout (seconds) |
| `max_retries` | `LANGEXTRACT_MAX_RETRIES` | integer | `3` | Maximum retry attempts |
| `use_parallel` | `LANGEXTRACT_USE_PARALLEL` | boolean | `false` | Use parallel processing |

### Processing Settings

| Setting | Environment Variable | Type | Default | Description |
|---------|---------------------|------|---------|-------------|
| `chunk_size` | `LANGEXTRACT_CHUNK_SIZE` | integer | `1000` | Text chunk size |
| `overlap_size` | `LANGEXTRACT_OVERLAP_SIZE` | integer | `100` | Chunk overlap size |
| `min_confidence` | `LANGEXTRACT_MIN_CONFIDENCE` | float | `0.6` | Minimum confidence threshold |

### Debug Settings

| Setting | Environment Variable | Type | Default | Description |
|---------|---------------------|------|---------|-------------|
| `debug_mode` | `LANGEXTRACT_DEBUG_MODE` | boolean | `false` | Enable debug logging |
| `log_level` | `LANGEXTRACT_LOG_LEVEL` | string | `"INFO"` | Logging level |

## Supported Models

LangExtract supports all models available in mLLMCelltype:

### Google/Gemini Models
- `gemini-2.5-flash` (recommended, fast and cost-effective)
- `gemini-2.5-pro` (higher accuracy)
- `gemini-2.0-flash`

### OpenAI Models
- `gpt-5`
- `gpt-5-mini`
- `gpt-4-turbo`

### Anthropic Models
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`

### Other Providers
- DeepSeek: `deepseek-chat`, `deepseek-r1`
- Qwen: `qwen-max-2025-01-25`
- And many more...

## API Key Management

### Environment Variables

Set the appropriate API key for your chosen model provider:

```bash
# Google/Gemini
GEMINI_API_KEY=your_key

# OpenAI
OPENAI_API_KEY=your_key

# Anthropic
ANTHROPIC_API_KEY=your_key

# DeepSeek
DEEPSEEK_API_KEY=your_key

# And so on...
```

### Getting API Keys

| Provider | URL | Notes |
|----------|-----|-------|
| Google/Gemini | https://aistudio.google.com/app/apikey | Free tier available |
| OpenAI | https://platform.openai.com/api-keys | Pay-per-use |
| Anthropic | https://console.anthropic.com/ | Claude models |
| DeepSeek | https://platform.deepseek.com/api_keys | Competitive pricing |

## Configuration Validation

The system automatically validates all configuration values:

- **Type checking**: Ensures values are of correct types
- **Range validation**: Checks numeric values are within valid ranges
- **Dependency validation**: Ensures related settings are compatible
- **API key validation**: Checks if required API keys are available

### Common Validation Issues

1. **Invalid complexity threshold**: Must be between 0.0 and 1.0
2. **Missing API key**: Required for the selected model
3. **Invalid timeout values**: Must be positive numbers
4. **Overlapping chunk sizes**: overlap_size must be < chunk_size

## Advanced Configuration

### Function-Level Overrides

You can override configuration at the function level:

```r
# R example
result <- some_langextract_function(
  langextract_enabled = TRUE,
  langextract_model = "gpt-4o",
  langextract_threshold = 0.8
)
```

```python
# Python example
result = some_langextract_function(
    langextract_enabled=True,
    langextract_model="gpt-4o",
    langextract_threshold=0.8
)
```

### Configuration Merging

Use the merge functions to combine configurations:

```r
# R
base_config <- load_langextract_config()
final_config <- merge_langextract_config(
  base_config,
  list(model = "claude-3-5-sonnet-20241022")
)
```

```python
# Python
from mllmcelltype.langextract_config import merge_langextract_config

base_config = load_langextract_config()
final_config = merge_langextract_config(
    base_config,
    {"model": "claude-3-5-sonnet-20241022"}
)
```

## Best Practices

### 1. Model Selection

- **For development/testing**: Use `gemini-2.5-flash` (fast, cheap)
- **For production**: Use `gemini-2.5-pro` or `gpt-4o` (higher accuracy)
- **For budget-conscious**: Use `gpt-4o-mini` or `deepseek-chat`

### 2. Complexity Threshold Tuning

- **Conservative (0.8-0.9)**: Only use LangExtract for very complex texts
- **Balanced (0.6-0.7)**: Good balance of accuracy and efficiency
- **Aggressive (0.3-0.5)**: Use LangExtract more frequently

### 3. Performance Optimization

```bash
# For better performance
LANGEXTRACT_CACHE_ENABLED=true
LANGEXTRACT_CHUNK_SIZE=1500
LANGEXTRACT_API_TIMEOUT=45
LANGEXTRACT_MAX_RETRIES=2
```

### 4. Development Settings

```bash
# For debugging
LANGEXTRACT_DEBUG_MODE=true
LANGEXTRACT_LOG_LEVEL=DEBUG
LANGEXTRACT_API_TIMEOUT=60
```

### 5. Production Settings

```bash
# For production
LANGEXTRACT_DEBUG_MODE=false
LANGEXTRACT_LOG_LEVEL=INFO
LANGEXTRACT_CACHE_ENABLED=true
LANGEXTRACT_FALLBACK_ENABLED=true
```

## Configuration Health Checks

Both R and Python provide health check functions:

```r
# R
health <- check_langextract_config_health(config)
if (health$status == "unhealthy") {
  print("Issues found:")
  print(health$issues)
}
```

```python
# Python
health = check_langextract_config_health(config)
if health["status"] == "unhealthy":
    print("Issues found:")
    for issue in health["issues"]:
        print(f"- {issue}")
```

## Troubleshooting

### Common Issues

1. **"No API key found"**
   - Solution: Set the appropriate `{PROVIDER}_API_KEY` environment variable

2. **"Unsupported model"**
   - Solution: Check the model name against supported models list

3. **"Configuration validation failed"**
   - Solution: Check the specific validation error and fix the invalid values

4. **"LangExtract disabled"**
   - Solution: Set `LANGEXTRACT_ENABLED=true`

### Debug Mode

Enable debug mode for detailed logging:

```bash
LANGEXTRACT_DEBUG_MODE=true
LANGEXTRACT_LOG_LEVEL=DEBUG
```

This will provide detailed information about:
- Configuration loading process
- API key resolution
- Validation steps
- Runtime decisions

## Integration Examples

### R Package Integration

```r
# In your R function
annotate_with_langextract <- function(data, ...) {
  # Load base configuration
  config <- load_langextract_config()
  
  # Merge with function parameters
  func_params <- list(...)
  final_config <- merge_langextract_config(config, func_params)
  
  # Check if LangExtract should be used
  if (!final_config$enabled) {
    return(standard_annotation(data))
  }
  
  # Use LangExtract
  return(langextract_annotation(data, final_config))
}
```

### Python Package Integration

```python
from mllmcelltype.langextract_config import (
    load_langextract_config,
    merge_langextract_config,
    should_use_langextract
)

def annotate_with_langextract(data, **kwargs):
    """Annotation function with LangExtract integration."""
    # Load configuration
    config = load_langextract_config()
    final_config = merge_langextract_config(config, kwargs)
    
    # Calculate complexity (example)
    complexity = calculate_complexity(data)
    
    # Decide whether to use LangExtract
    if should_use_langextract(complexity, final_config):
        return langextract_annotation(data, final_config)
    else:
        return standard_annotation(data)
```

This configuration system provides a robust foundation for LangExtract integration while maintaining flexibility and ease of use across both R and Python implementations.