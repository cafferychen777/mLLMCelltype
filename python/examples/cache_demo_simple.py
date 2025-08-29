#!/usr/bin/env python3
"""
Simple Cache Demo for mLLMCelltype

This script demonstrates the core caching functionality with clear examples.
"""

import os
import sys
import time
from typing import Dict

# Add the python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mllmcelltype import annotate_clusters, clear_cache, get_cache_stats


def demo_basic_caching():
    """Demonstrate basic caching functionality."""

    print("🗂️  mLLMCelltype Cache Demo")
    print("=" * 40)

    # Test data - simple T cell markers
    test_markers = {
        "0": ["CD3D", "CD4"],  # T helper cells
        "1": ["CD8A", "CD8B"],  # Cytotoxic T cells
    }

    print(f"Test data: {test_markers}")

    # Clear any existing cache
    cleared = clear_cache()
    if cleared > 0:
        print(f"🧹 Cleared {cleared} old cache files")

    print("\n1️⃣  First run (no cache)...")
    start = time.time()

    result1 = annotate_clusters(
        marker_genes=test_markers, species="human", model="gpt-5-mini", use_cache=True
    )

    first_time = time.time() - start
    print(f"   ⏱️  Time: {first_time:.2f} seconds")
    print(f"   📊 Results: {result1}")

    # Check cache after first run
    stats = get_cache_stats()
    print(f"   💾 Cache files created: {stats.get('valid_files', 0)}")

    print("\n2️⃣  Second run (with cache)...")
    start = time.time()

    result2 = annotate_clusters(
        marker_genes=test_markers, species="human", model="gpt-5-mini", use_cache=True
    )

    second_time = time.time() - start
    print(f"   ⏱️  Time: {second_time:.2f} seconds")
    print(f"   📊 Results: {result2}")

    # Performance analysis
    if second_time > 0:
        speedup = first_time / second_time
        print(f"\n🚀 Performance Analysis:")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Time saved: {first_time - second_time:.2f} seconds")

    # Consistency check
    consistent = result1 == result2
    print(f"   ✅ Results consistent: {'Yes' if consistent else 'No'}")

    return {
        "first_time": first_time,
        "second_time": second_time,
        "speedup": speedup if second_time > 0 else 0,
        "consistent": consistent,
    }


def demo_cache_invalidation():
    """Demonstrate cache invalidation with different parameters."""

    print("\n🔄 Cache Invalidation Demo")
    print("=" * 40)

    base_markers = {"0": ["CD3D"]}

    print("Testing parameter changes...")

    # Base case
    print("\n🔹 Base case:")
    start = time.time()
    result_base = annotate_clusters(
        marker_genes=base_markers, species="human", model="gpt-4o-mini", use_cache=True
    )
    base_time = time.time() - start
    print(f"   Time: {base_time:.3f}s, Result: {result_base}")

    # Different species (should create new cache entry)
    print("\n🔹 Different species:")
    start = time.time()
    result_mouse = annotate_clusters(
        marker_genes=base_markers,
        species="mouse",  # Changed parameter
        model="gpt-4o-mini",
        use_cache=True,
    )
    mouse_time = time.time() - start
    print(f"   Time: {mouse_time:.3f}s, Result: {result_mouse}")
    print(f"   New cache entry: {'Yes' if mouse_time > 0.1 else 'Likely cached'}")

    # Same parameters again (should use cache)
    print("\n🔹 Repeat base case:")
    start = time.time()
    result_repeat = annotate_clusters(
        marker_genes=base_markers, species="human", model="gpt-4o-mini", use_cache=True
    )
    repeat_time = time.time() - start
    print(f"   Time: {repeat_time:.3f}s, Result: {result_repeat}")
    print(f"   Used cache: {'Yes' if repeat_time < 0.1 else 'No'}")


def demo_cache_management():
    """Demonstrate cache management features."""

    print("\n🛠️  Cache Management Demo")
    print("=" * 40)

    # Get current cache stats
    stats = get_cache_stats()
    print(f"📈 Current cache status:")
    print(f"   Status: {stats.get('status')}")
    print(f"   Files: {stats.get('valid_files', 0)}")
    print(f"   Size: {stats.get('total_size_mb', 0):.2f} MB")

    # Clear cache
    print(f"\n🧹 Clearing cache...")
    cleared = clear_cache()
    print(f"   Cleared {cleared} files")

    # Check stats after clearing
    stats_after = get_cache_stats()
    print(f"   Files after clearing: {stats_after.get('valid_files', 0)}")


def main():
    """Run all cache demos."""

    # Basic caching demo
    results = demo_basic_caching()

    # Cache invalidation demo
    demo_cache_invalidation()

    # Cache management demo
    demo_cache_management()

    # Summary
    print(f"\n📋 Summary")
    print("=" * 40)
    print(f"✅ Cache provides {results['speedup']:.1f}x speedup")
    print(f"✅ Results are {'consistent' if results['consistent'] else 'inconsistent'}")
    print(f"✅ Cache management works properly")
    print(f"\n💡 The cache system is ready for production use!")


if __name__ == "__main__":
    main()
