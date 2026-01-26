"""mLLMCelltype: A Python module for cell type annotation using various LLMs."""

from .annotate import annotate_clusters, get_model_response
from .consensus import (
    check_consensus,
    format_discussion_report,
    interactive_consensus_annotation,
    process_controversial_clusters,
)
from .functions import get_provider
from .logger import setup_logging, write_log
from .prompts import (
    create_consensus_check_prompt,
    create_discussion_prompt,
    create_prompt,
)
from .url_utils import (
    get_default_api_url,
    resolve_provider_base_url,
    validate_base_url,
)
from .utils import (
    clean_annotation,
    clear_cache,
    create_cache_key,
    format_results,
    get_cache_stats,
    load_api_key,
    load_from_cache,
    save_to_cache,
)

__version__ = "2.0.1"

__all__ = [
    # Core annotation
    "annotate_clusters",
    "get_model_response",
    # Functions
    "get_provider",
    "clean_annotation",
    # Logging
    "setup_logging",
    "write_log",
    # Utils
    "load_api_key",
    "create_cache_key",
    "save_to_cache",
    "load_from_cache",
    "clear_cache",
    "get_cache_stats",
    "format_results",
    # Prompts
    "create_prompt",
    "create_discussion_prompt",
    "create_consensus_check_prompt",
    # Consensus
    "check_consensus",
    "process_controversial_clusters",
    "interactive_consensus_annotation",
    "format_discussion_report",
    # URL utilities
    "resolve_provider_base_url",
    "get_default_api_url",
    "validate_base_url",
]
