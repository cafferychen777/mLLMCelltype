"""Cache management module for mLLMCelltype.

This module provides utilities for managing the mLLMCelltype cache system,
including cache inspection, clearing, and validation functions.

Functions:
    clear_mllmcelltype_cache(): Interactive cache clearing
    get_cache_info(): Get information about current cache
    clear_cache_cli(): Command-line interface for cache management
"""

import os
import shutil


def clear_mllmcelltype_cache():
    """Clear the mLLMCelltype cache directory."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".llmcelltype", "cache")

    if os.path.exists(cache_dir):
        print(f"Found cache directory: {cache_dir}")

        # Count cache files
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
        print(f"Found {len(cache_files)} cache files")

        # Ask for confirmation
        response = input("Do you want to clear all cache files? (yes/no): ")
        if response.lower() == "yes":
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print("Cache cleared successfully!")
        else:
            print("Cache clearing cancelled.")
    else:
        print("No cache directory found.")


def get_cache_info():
    """Get information about the current cache state."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".llmcelltype", "cache")

    if not os.path.exists(cache_dir):
        return {"exists": False, "path": cache_dir, "file_count": 0, "total_size": 0}

    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)

    return {
        "exists": True,
        "path": cache_dir,
        "file_count": len(cache_files),
        "total_size": total_size,
        "size_mb": total_size / (1024 * 1024),
    }


def clear_cache_cli():
    """Command-line interface for cache management."""
    import sys

    print("mLLMCelltype Cache Manager")
    print("-" * 30)

    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        # Non-interactive mode
        from .utils import clear_cache

        removed = clear_cache()
        print(f"\nCleared {removed} cache files.")
    elif len(sys.argv) > 1 and sys.argv[1] == "--info":
        # Show cache info
        info = get_cache_info()
        print(f"\nCache directory: {info['path']}")
        print(f"Number of cache files: {info['file_count']}")
        print(f"Total cache size: {info['size_mb']:.2f} MB")
    else:
        # Interactive mode
        clear_mllmcelltype_cache()

    print("\nUsage:")
    print("  python -m mllmcelltype.cache_manager          # Interactive mode")
    print("  python -m mllmcelltype.cache_manager --clear  # Clear cache without confirmation")
    print("  python -m mllmcelltype.cache_manager --info   # Show cache information")


if __name__ == "__main__":
    clear_cache_cli()
