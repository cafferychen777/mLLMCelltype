# mLLMCelltype Cache System Documentation

## Problem Description

Users encountered cache issues when switching between different models using the `interactive_consensus_annotation` function, specifically:
- After switching to OpenRouter models, the system still returned cached results from previous models
- Different API calls failed, but the system continued using old cached data
- Cache keys did not properly distinguish between different providers

## Root Cause

The cache key generation did not properly handle the special case of OpenRouter models. OpenRouter models use the "provider/model" format (e.g., "openai/gpt-5-mini"), but the cache system might use incorrect provider information when generating cache keys.

## Implemented Long-term Solution

### 1. Modified Cache Key Generation Logic (utils.py)

```python
def create_cache_key(prompt: str, model: str, provider: str) -> str:
    # Normalize inputs to ensure consistent keys
    normalized_provider = str(provider).lower().strip()
    normalized_model = str(model).lower().strip()
    normalized_prompt = str(prompt).strip()

    # For OpenRouter models (containing '/'), ensure provider is 'openrouter'
    # This prevents cache key collisions between different providers
    if "/" in normalized_model:
        normalized_provider = "openrouter"

    # Create a string to hash with clear separators to avoid collisions
    hash_string = (
        f"provider:{normalized_provider}||model:{normalized_model}||prompt:{normalized_prompt}"
    )

    # Create hash
    hash_object = hashlib.sha256(hash_string.encode("utf-8"))
    return hash_object.hexdigest()
```

### 2. Test Validation

Created a comprehensive test suite `tests/test_cache_system.py` to verify:
- ✅ Cache keys correctly include provider information
- ✅ OpenRouter models always use "openrouter" as provider
- ✅ Same model from different providers do not share cache
- ✅ Real-world scenario tests pass

### 3. Cache Management Module

Provided `cache_manager` module supporting:
- View cache info: `python -m mllmcelltype.cache_manager --info`
- Clear cache: `python -m mllmcelltype.cache_manager --clear`
- Interactive management: `python -m mllmcelltype.cache_manager`

Also available in code:
```python
from mllmcelltype import get_cache_info, clear_cache

# Get cache information
info = get_cache_info()
print(f"Cache files: {info['file_count']}")
print(f"Cache size: {info['size_mb']:.2f} MB")

# Clear cache
removed = clear_cache()
print(f"Removed {removed} cache files")
```

## Usage Recommendations

### 1. Clear Cache After Code Update
```bash
python -m mllmcelltype.cache_manager --clear
```

### 2. Properly Specify Models
```python
# OpenRouter models
models = [
    {"provider": "openrouter", "model": "openai/gpt-5-mini"},
    {"provider": "openrouter", "model": "anthropic/claude-3-opus"},
]

# Or let the system auto-detect (recommended)
models = [
    "openai/gpt-4o-mini",  # Auto-detected as openrouter
    "anthropic/claude-3-opus",  # Auto-detected as openrouter
    "gpt-4o",  # Auto-detected as openai
    "claude-3-opus",  # Auto-detected as anthropic
]
```

### 3. Temporarily Disable Cache (if needed)
```python
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="Hepatocellular carcinoma",
    models=models,
    use_cache=False  # Temporarily disable cache
)
```

## Technical Details

### Cache Key Components
- Provider (normalized provider name)
- Model (normalized model name)
- Prompt (user-provided prompt)

### Special Handling
- Model names containing "/" are automatically classified as OpenRouter
- All inputs are normalized to lowercase and trimmed
- SHA256 is used to generate unique cache keys

### Cache Storage Location
- Default: `~/.llmcelltype/cache/`
- Format: JSON files containing version and timestamp

## Monitoring and Maintenance

### View Cache Statistics
```python
from mllmcelltype.utils import get_cache_stats
stats = get_cache_stats()
print(f"Cache files: {stats['count']}")
print(f"Cache size: {stats['size_readable']}")
```

### Periodic Cache Cleanup
```python
from mllmcelltype.utils import clear_cache
# Clear cache older than 7 days
removed = clear_cache(older_than=7*24*60*60)
print(f"Removed {removed} cache files")
```

## Verify Fix Effectiveness

Run the test suite to confirm the fix is working:
```bash
cd tests
python test_cache_system.py
```

Expected output:
```
✅ ALL TESTS PASSED!
The cache fix is working correctly.
```

## Future Recommendations

1. **Version Control**: Consider cache versioning in future releases for automatic cleanup of incompatible caches during upgrades
2. **Cache Expiration**: Consider adding cache expiration time settings
3. **Cache Size Limits**: Consider adding cache size limits with automatic cleanup of oldest entries

## Related Files

- `/mllmcelltype/utils.py` - Contains the fixed cache key generation logic
- `/mllmcelltype/cache_manager.py` - Cache management module
- `/tests/test_cache_system.py` - Comprehensive test suite
- `/docs/CACHE_SYSTEM.md` - This document

## Issue Tracking

GitHub Issue: https://github.com/cafferychen777/mLLMCelltype/issues/65