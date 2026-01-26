"""Cache management module for mLLMCelltype.

This module provides utilities for managing the mLLMCelltype cache system,
including cache inspection, clearing, and validation functions.

Functions:
    clear_mllmcelltype_cache(): Interactive cache clearing
    clear_cache_cli(): Command-line interface for cache management
"""

import os
import shutil

from .utils import get_cache_stats


def clear_mllmcelltype_cache():
    """Clear the mLLMCelltype cache directory."""
    info = get_cache_stats(detailed=False)
    cache_dir = info["path"]

    if info["exists"]:
        print(f"Found cache directory: {cache_dir}")
        print(f"Found {info['count']} cache files")

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
        info = get_cache_stats(detailed=False)
        print(f"\nCache directory: {info['path']}")
        print(f"Number of cache files: {info['count']}")
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
