"""mLLMCelltype: A Python module for cell type annotation using various LLMs."""

from .annotate import annotate_clusters, get_model_response
from .config import (
    DEFAULT_CONSENSUS_CONFIG,
    PROVIDER_CONFIGS,
    get_api_key_env_var,
    get_default_api_url,
    get_default_model,
    get_supported_providers,
)
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

__version__ = "2.0.2"

__all__ = [
    "DEFAULT_CONSENSUS_CONFIG",
    "PROVIDER_CONFIGS",
    "annotate_clusters",
    "check_consensus",
    "clean_annotation",
    "clear_cache",
    "create_cache_key",
    "create_consensus_check_prompt",
    "create_discussion_prompt",
    "create_prompt",
    "format_discussion_report",
    "format_results",
    "get_api_key_env_var",
    "get_cache_stats",
    "get_default_api_url",
    "get_default_model",
    "get_model_response",
    "get_provider",
    "get_supported_providers",
    "interactive_consensus_annotation",
    "load_api_key",
    "load_from_cache",
    "process_controversial_clusters",
    "resolve_provider_base_url",
    "save_to_cache",
    "setup_logging",
    "validate_base_url",
    "write_log",
]
