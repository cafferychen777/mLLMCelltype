"""Verify annotation storage in the production container without API calls."""

from pathlib import Path
from uuid import uuid4

from mllmcelltype import annotate_clusters, get_current_log_file
from mllmcelltype.utils import load_from_cache, save_to_cache


def main():
    # Empty input exercises logging initialization without contacting a provider.
    assert annotate_clusters({}, species="human", provider="zhipu") == {}
    log_file = get_current_log_file()
    assert log_file is not None and Path(log_file).is_file()

    cache_key = f"runtime-smoke-{uuid4().hex}"
    cache_file = Path.home() / ".mllmcelltype" / "cache" / f"{cache_key}.json"
    try:
        save_to_cache(cache_key, {"0": "T cells"})
        assert load_from_cache(cache_key) == {"0": "T cells"}
    finally:
        cache_file.unlink(missing_ok=True)
    print("Annotation log and cache storage checks passed.")


if __name__ == "__main__":
    main()
