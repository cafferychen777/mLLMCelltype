"""mLLMCelltype: A Python module for cell type annotation using various LLMs."""

from .annotate import annotate_clusters, batch_annotate_clusters, get_model_response
from .cache_manager import get_cache_info
from .consensus import (
    check_consensus,
    facilitate_cluster_discussion,
    interactive_consensus_annotation,
    print_consensus_summary,
    process_controversial_clusters,
    summarize_discussion,
)
from .functions import (
    get_provider,
    identify_controversial_clusters,
    select_best_prediction,
)
from .logger import setup_logging, write_log
from .prompts import (
    create_batch_prompt,
    create_consensus_check_prompt,
    create_discussion_prompt,
    create_initial_discussion_prompt,
    create_json_prompt,
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
    find_agreement,
    format_results,
    get_annotation_metadata,
    get_cache_stats,
    load_api_key,
    load_from_cache,
    save_to_cache,
    validate_cache,
)

# LangExtract components (simplified)

# LangExtract parser module (optional dependency)
try:
    from .langextract_parser import (
        BatchAnnotationResult,
        CellTypeAnnotation,
        ConsensusMetrics,
        DiscussionAnalysis,
        LangextractParser,
        ParsingComplexity,
        ParsingConfig,
        analyze_consensus,
        create_parser,
        parse_cell_types,
    )

    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False

__version__ = "1.3.3"

__all__ = [
    # Core annotation
    "annotate_clusters",
    "batch_annotate_clusters",
    "get_model_response",
    # Functions
    "get_provider",
    "clean_annotation",
    "identify_controversial_clusters",
    "select_best_prediction",
    # Logging
    "setup_logging",
    "write_log",
    # Utils
    "load_api_key",
    "create_cache_key",
    "save_to_cache",
    "load_from_cache",
    "validate_cache",
    "clear_cache",
    "get_cache_stats",
    "get_cache_info",
    "get_annotation_metadata",
    "format_results",
    "find_agreement",
    # Prompts
    "create_prompt",
    "create_batch_prompt",
    "create_json_prompt",
    "create_discussion_prompt",
    "create_consensus_check_prompt",
    "create_initial_discussion_prompt",
    # Consensus
    "check_consensus",
    "process_controversial_clusters",
    "interactive_consensus_annotation",
    "print_consensus_summary",
    "facilitate_cluster_discussion",
    "summarize_discussion",
    # URL utilities
    "resolve_provider_base_url",
    "get_default_api_url",
    "validate_base_url",
    # LangExtract configuration
    "get_default_langextract_config",
    "load_langextract_config",
    "validate_langextract_config",
    "merge_langextract_config",
    "get_langextract_api_key",
    "print_langextract_config",
    "check_langextract_config_health",
    "get_environment_variables_template",
    "is_langextract_enabled",
    "get_langextract_model",
    "should_use_langextract",
]

# Add LangExtract functionality if available
if LANGEXTRACT_AVAILABLE:
    __all__.extend(
        [
            # LangExtract Parser
            "LangextractParser",
            "ParsingConfig",
            "CellTypeAnnotation",
            "ConsensusMetrics",
            "BatchAnnotationResult",
            "DiscussionAnalysis",
            "ParsingComplexity",
            "create_parser",
            "parse_cell_types",
            "analyze_consensus",
            "LANGEXTRACT_AVAILABLE",
        ]
    )
