#!/usr/bin/env python3
"""
Example of cache management in mLLMCelltype.

This example shows how to:
1. Check cache status
2. Clear cache programmatically
3. Use cache effectively with different models
"""

from mllmcelltype import (
    get_cache_info,
    get_cache_stats,
)


def demonstrate_cache_management():
    """Demonstrate cache management features."""

    print("=== mLLMCelltype Cache Management Example ===\n")

    # 1. Check current cache status
    print("1. Checking cache status...")
    info = get_cache_info()
    if info["exists"]:
        print(f"   Cache directory: {info['path']}")
        print(f"   Number of files: {info['file_count']}")
        print(f"   Total size: {info['size_mb']:.2f} MB")
    else:
        print("   No cache directory found")

    # 2. Get detailed cache statistics
    print("\n2. Getting cache statistics...")
    stats = get_cache_stats()
    print(f"   Status: {stats['status']}")
    print(f"   Valid files: {stats.get('valid_files', 0)}")
    print(f"   Format distribution: {stats.get('format_counts', {})}")

    # 3. Clear old cache entries (optional)
    print("\n3. Cache management options:")
    print("   - To clear all cache: clear_cache()")
    print("   - To clear old cache: clear_cache(older_than=7*24*60*60)  # 7 days")
    print("   - To disable cache: use use_cache=False in function calls")


def demonstrate_proper_model_usage():
    """Demonstrate proper model specification to avoid cache issues."""

    print("\n=== Proper Model Usage ===\n")

    # Example marker genes (just for demonstration purposes)
    # marker_genes = {
    #     "0": ["CD3D", "CD3E", "CD4", "IL7R"],
    #     "1": ["CD8A", "CD8B", "GZMK", "CCL5"],
    #     "2": ["MS4A1", "CD79A", "CD79B", "CD19"],
    # }

    print("1. Using regular models (auto-detected providers):")
    models_regular = [
        "gpt-4o",  # OpenAI
        "claude-3-opus",  # Anthropic
        "qwen-max-2025-01-25",  # Qwen
    ]
    print(f"   Models: {models_regular}")

    print("\n2. Using OpenRouter models (auto-detected as openrouter):")
    models_openrouter = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-opus",
        "meta-llama/llama-3.1-405b-instruct",
    ]
    print(f"   Models: {models_openrouter}")

    print("\n3. Explicitly specifying providers (optional):")
    models_explicit = [
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
        {"provider": "openrouter", "model": "anthropic/claude-3-opus"},
    ]
    print("   Models with explicit providers:")
    for m in models_explicit:
        print(f"     - {m}")

    print("\n4. Example usage with cache control:")
    print("""
    # Enable cache (default)
    results = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human",
        models=models_openrouter,
        use_cache=True  # Default
    )
    
    # Disable cache for testing
    results = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human",
        models=models_openrouter,
        use_cache=False  # Bypass cache
    )
    """)


def main():
    """Main example function."""

    # Demonstrate cache management
    demonstrate_cache_management()

    # Demonstrate proper model usage
    demonstrate_proper_model_usage()

    print("\n=== Summary ===")
    print("1. The cache system now properly handles OpenRouter models")
    print("2. Models with '/' are automatically detected as OpenRouter")
    print("3. Use cache_manager module for cache management")
    print("4. Run tests/test_cache_system.py to verify cache behavior")


if __name__ == "__main__":
    main()
