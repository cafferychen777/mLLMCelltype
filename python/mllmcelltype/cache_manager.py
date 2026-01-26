"""Cache management CLI for mLLMCelltype.

Usage:
    python -m mllmcelltype.cache_manager          # Interactive mode
    python -m mllmcelltype.cache_manager --clear  # Clear cache without confirmation
    python -m mllmcelltype.cache_manager --info   # Show cache information
"""

from .utils import clear_cache, get_cache_stats


def clear_cache_cli():
    """Command-line interface for cache management."""
    import sys

    print("mLLMCelltype Cache Manager")
    print("-" * 30)

    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        removed = clear_cache()
        print(f"\nCleared {removed} cache files.")

    elif len(sys.argv) > 1 and sys.argv[1] == "--info":
        info = get_cache_stats(detailed=False)
        print(f"\nCache directory: {info['path']}")
        print(f"Number of cache files: {info['count']}")
        print(f"Total cache size: {info['size_mb']:.2f} MB")

    else:
        # Interactive mode
        info = get_cache_stats(detailed=False)
        if info["exists"]:
            print(f"Found cache directory: {info['path']}")
            print(f"Found {info['count']} cache files")
            response = input("\nDo you want to clear all cache files? (yes/no): ")
            if response.lower() == "yes":
                removed = clear_cache()
                print(f"Cleared {removed} cache files.")
            else:
                print("Cache clearing cancelled.")
        else:
            print("No cache directory found.")

    print("\nUsage:")
    print("  python -m mllmcelltype.cache_manager          # Interactive mode")
    print("  python -m mllmcelltype.cache_manager --clear  # Clear without confirmation")
    print("  python -m mllmcelltype.cache_manager --info   # Show cache information")


if __name__ == "__main__":
    clear_cache_cli()
