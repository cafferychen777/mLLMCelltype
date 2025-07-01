Hi @eason-analytics,

Thank you for reporting this issue! I've identified and fixed the cache retention problem you encountered.

## Root Cause

The issue was in the cache key generation system. When switching between regular models (like `gpt-4o`) and OpenRouter models (like `openai/gpt-4o-mini`), the cache keys weren't properly isolated because OpenRouter models weren't normalizing their provider correctly. This caused OpenRouter models to incorrectly reuse cached results from regular models.

## The Fix

I've updated the `create_cache_key` function in `utils.py` to properly handle OpenRouter models:

```python
def create_cache_key(prompt: str, model: str, provider: str) -> str:
    # ... existing normalization code ...
    
    # For OpenRouter models (containing '/'), ensure provider is 'openrouter'
    # This prevents cache key collisions between different providers
    if "/" in normalized_model:
        normalized_provider = "openrouter"
    
    # ... rest of the function ...
```

This ensures that:
1. OpenRouter models (identified by "/" in the model name) always use "openrouter" as the provider in cache keys
2. Different providers maintain separate cache entries
3. No cache collision occurs between similar model names from different providers

## Verifying the Fix

To verify the fix is working in your environment:

1. **Update to the latest version** (v1.2.9 or later)
   ```bash
   pip install --upgrade mllmcelltype
   ```

2. **Clear your existing cache** to remove any problematic entries:
   ```python
   from mllmcelltype import clear_cache
   clear_cache()
   ```
   
   Or use the CLI:
   ```bash
   python -m mllmcelltype.cache_manager --clear
   ```

3. **Check cache status**:
   ```python
   from mllmcelltype import get_cache_info
   info = get_cache_info()
   print(f"Cache files: {info['file_count']}")
   print(f"Cache size: {info['size_mb']:.2f} MB")
   ```

4. **Run your analysis** - the cache will now properly isolate different providers:
   ```python
   # These will use separate cache entries:
   models_regular = ["gpt-4o", "claude-3-opus"]
   models_openrouter = ["openai/gpt-4o-mini", "anthropic/claude-3-opus"]
   ```

## Additional Notes

- This was a universal issue affecting all models, not just Qwen (the 401 error you saw with Qwen was an authentication issue)
- The cache system now properly detects OpenRouter models by the "/" in their name
- You can disable caching for testing with `use_cache=False` in function calls

I've also added comprehensive tests to ensure this issue doesn't recur. You can run them yourself:
```bash
cd python
python test_cache_system.py
```

Please let me know if you encounter any other issues after updating!