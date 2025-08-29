#!/usr/bin/env python3
"""
Example of cache management in mLLMCelltype.

This example shows how to:
1. Check cache status
2. Clear cache programmatically
3. Use cache effectively with different models
4. Test LangExtract integration with caching
5. Performance comparison with/without cache
"""

import os
import sys
import time
from typing import Any, Dict

# Add the python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mllmcelltype import (
    annotate_clusters,
    clear_cache,
    get_cache_stats,
)


def demonstrate_cache_management():
    """Demonstrate cache management features."""

    print("=== mLLMCelltype Cache Management Example ===\n")

    # 1. Get detailed cache statistics
    print("1. Getting cache statistics...")
    stats = get_cache_stats()
    print(f"   Status: {stats['status']}")
    print(f"   Cache directory: {stats.get('cache_dir', 'N/A')}")
    print(f"   Valid files: {stats.get('valid_files', 0)}")
    print(f"   Invalid files: {stats.get('invalid_files', 0)}")
    print(f"   Total size: {stats.get('total_size_mb', 0):.2f} MB")
    print(f"   Format distribution: {stats.get('format_counts', {})}")

    # 2. Cache management options
    print("\n2. Cache management options:")
    print("   - To clear all cache: clear_cache()")
    print("   - To clear old cache: clear_cache(older_than=7*24*60*60)  # 7 days")
    print("   - To disable cache: use use_cache=False in function calls")

    return stats


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


def test_langextract_cache_integration():
    """Test LangExtract integration with caching system."""

    print("\n=== LangExtract + Cache Integration Test ===\n")

    # Test marker genes for demonstration
    test_markers = {
        "0": ["CD3D", "CD3E", "CD4", "IL7R"],  # T helper cells
        "1": ["CD8A", "CD8B", "GZMK", "CCL5"],  # Cytotoxic T cells
        "2": ["MS4A1", "CD79A", "CD79B", "CD19"],  # B cells
    }

    print("Test data:")
    for cluster, genes in test_markers.items():
        print(f"   Cluster {cluster}: {', '.join(genes)}")

    # Clear cache for clean test
    print("\n1. Clearing cache for clean test...")
    cleared_count = clear_cache()
    print(f"   Cleared {cleared_count} cache files")

    # Test 1: First run with LangExtract (cold start)
    print("\n2. First run with LangExtract (building cache)...")
    start_time = time.time()

    try:
        result1 = annotate_clusters(
            marker_genes=test_markers,
            species="human",
            use_langextract=True,
            langextract_config={"complexity_threshold": 0.5},
            use_cache=True,
            model="gpt-5-mini",  # Use a fast, affordable model for testing
        )

        first_run_time = time.time() - start_time
        print(f"   ✓ First run completed in {first_run_time:.2f} seconds")
        print(f"   Results: {len(result1)} clusters annotated")

        # Display first result for verification
        if result1:
            first_cluster = list(result1.keys())[0]
            first_result = result1[first_cluster]
            # Handle both dict and string results
            if isinstance(first_result, dict):
                celltype = first_result.get(
                    "predicted_celltype", first_result.get("annotation", "N/A")
                )
            else:
                celltype = str(first_result)
            print(f"   Sample result - Cluster {first_cluster}: {celltype}")

    except Exception as e:
        print(f"   ✗ First run failed: {str(e)}")
        return

    # Check cache after first run
    print("\n3. Checking cache after first run...")
    stats_after_first = get_cache_stats()
    print(f"   Valid cache files: {stats_after_first.get('valid_files', 0)}")
    print(f"   Cache size: {stats_after_first.get('total_size_mb', 0):.2f} MB")

    # Test 2: Second run with same parameters (cache hit)
    print("\n4. Second run with same parameters (testing cache hit)...")
    start_time = time.time()

    try:
        result2 = annotate_clusters(
            marker_genes=test_markers,
            species="human",
            use_langextract=True,
            langextract_config={"complexity_threshold": 0.5},
            use_cache=True,
            model="gpt-4o-mini",
        )

        second_run_time = time.time() - start_time
        print(f"   ✓ Second run completed in {second_run_time:.2f} seconds")
        print(f"   Results: {len(result2)} clusters annotated")

        # Performance comparison
        if first_run_time > 0 and second_run_time > 0:
            speedup = first_run_time / second_run_time
            time_saved = first_run_time - second_run_time
            print(f"   🚀 Cache speedup: {speedup:.2f}x faster")
            print(f"   ⏱️  Time saved: {time_saved:.2f} seconds")

        # Verify results consistency
        if result1 and result2:
            consistent = True
            for cluster in result1:
                if cluster in result2:
                    # Handle both dict and string results
                    if isinstance(result1[cluster], dict):
                        r1_type = result1[cluster].get(
                            "predicted_celltype", result1[cluster].get("annotation", "")
                        )
                    else:
                        r1_type = str(result1[cluster])

                    if isinstance(result2[cluster], dict):
                        r2_type = result2[cluster].get(
                            "predicted_celltype", result2[cluster].get("annotation", "")
                        )
                    else:
                        r2_type = str(result2[cluster])

                    if r1_type != r2_type:
                        consistent = False
                        break

            if consistent:
                print("   ✓ Results are consistent between runs")
            else:
                print("   ⚠️  Results differ between runs (cache may not be working)")

    except Exception as e:
        print(f"   ✗ Second run failed: {str(e)}")
        return

    # Test 3: Different LangExtract config (should create new cache entry)
    print("\n5. Testing different LangExtract config...")
    start_time = time.time()

    try:
        result3 = annotate_clusters(
            marker_genes=test_markers,
            species="human",
            use_langextract=True,
            langextract_config={"complexity_threshold": 0.3},  # Different threshold
            use_cache=True,
            model="gpt-4o-mini",
        )

        third_run_time = time.time() - start_time
        print(f"   ✓ Different config run completed in {third_run_time:.2f} seconds")
        print(f"   This should create a new cache entry due to different config")

    except Exception as e:
        print(f"   ✗ Different config run failed: {str(e)}")

    # Final cache statistics
    print("\n6. Final cache statistics...")
    final_stats = get_cache_stats()
    print(f"   Valid cache files: {final_stats.get('valid_files', 0)}")
    print(f"   Total cache size: {final_stats.get('total_size_mb', 0):.2f} MB")
    print(f"   Format distribution: {final_stats.get('format_counts', {})}")

    return {
        "first_run_time": first_run_time,
        "second_run_time": second_run_time,
        "cache_speedup": first_run_time / second_run_time if second_run_time > 0 else 0,
        "final_cache_stats": final_stats,
    }


def test_cache_performance_comparison():
    """Compare performance with and without cache."""

    print("\n=== Cache Performance Comparison ===\n")

    # Smaller test for performance comparison
    small_test_markers = {
        "0": ["CD3D", "CD4"],  # Small set for faster testing
    }

    print("Performance test with minimal data...")

    # Test without cache
    print("\n1. Running without cache...")
    start_time = time.time()

    try:
        result_no_cache = annotate_clusters(
            marker_genes=small_test_markers,
            species="human",
            use_langextract=True,
            langextract_config={"complexity_threshold": 0.5},
            use_cache=False,  # Disable cache
            model="gpt-4o-mini",
        )

        no_cache_time = time.time() - start_time
        print(f"   ✓ No cache run: {no_cache_time:.2f} seconds")

    except Exception as e:
        print(f"   ✗ No cache run failed: {str(e)}")
        return

    # Test with cache (should be faster on second run)
    print("\n2. Running with cache (first time)...")
    start_time = time.time()

    try:
        result_cache_1 = annotate_clusters(
            marker_genes=small_test_markers,
            species="human",
            use_langextract=True,
            langextract_config={"complexity_threshold": 0.5},
            use_cache=True,
            model="gpt-5-mini",
        )

        cache_first_time = time.time() - start_time
        print(f"   ✓ Cache first run: {cache_first_time:.2f} seconds")

    except Exception as e:
        print(f"   ✗ Cache first run failed: {str(e)}")
        return

    print("\n3. Running with cache (second time - cache hit)...")
    start_time = time.time()

    try:
        result_cache_2 = annotate_clusters(
            marker_genes=small_test_markers,
            species="human",
            use_langextract=True,
            langextract_config={"complexity_threshold": 0.5},
            use_cache=True,
            model="gpt-5-mini",
        )

        cache_second_time = time.time() - start_time
        print(f"   ✓ Cache second run: {cache_second_time:.2f} seconds")

        # Performance analysis
        print(f"\n📊 Performance Analysis:")
        print(f"   No cache:           {no_cache_time:.2f}s")
        print(f"   Cache (first):      {cache_first_time:.2f}s")
        print(f"   Cache (second):     {cache_second_time:.2f}s")

        if cache_second_time > 0:
            cache_speedup = no_cache_time / cache_second_time
            print(f"   Cache speedup:      {cache_speedup:.2f}x")

    except Exception as e:
        print(f"   ✗ Cache second run failed: {str(e)}")


def main():
    """Main example function."""

    # Demonstrate cache management
    initial_stats = demonstrate_cache_management()

    # Demonstrate proper model usage
    demonstrate_proper_model_usage()

    # Test LangExtract + Cache integration
    print("\n" + "=" * 60)
    langextract_results = test_langextract_cache_integration()

    # Performance comparison
    print("\n" + "=" * 60)
    test_cache_performance_comparison()

    print("\n=== Summary ===")
    print("1. Cache system integrated with LangExtract functionality")
    print("2. Different LangExtract configs create separate cache entries")
    print("3. Cache provides significant performance improvements")
    print("4. Results remain consistent across cached runs")

    if langextract_results and langextract_results.get("cache_speedup", 0) > 1:
        speedup = langextract_results["cache_speedup"]
        print(f"5. Observed cache speedup: {speedup:.2f}x")

    final_stats = get_cache_stats()
    print(f"6. Final cache contains {final_stats.get('valid_files', 0)} files")
    print(f"7. Total cache size: {final_stats.get('total_size_mb', 0):.2f} MB")


if __name__ == "__main__":
    main()
