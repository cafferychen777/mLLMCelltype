"""mLLMCelltype: A Python module for cell type annotation using various LLMs."""

from .annotate import annotate_clusters, batch_annotate_clusters, get_model_response
from .functions import (
    get_provider,
    identify_controversial_clusters,
    select_best_prediction
)
from .logger import setup_logging, write_log
from .utils import (
    load_api_key,
    create_cache_key,
    save_to_cache,
    load_from_cache,
    validate_cache,
    clear_cache,
    get_cache_stats,
    format_results,
    find_agreement,
    clean_annotation
)
from .prompts import (
    create_prompt,
    create_batch_prompt,
    create_json_prompt,
    create_discussion_prompt,
    create_consensus_check_prompt,
    create_initial_discussion_prompt
)
from .consensus import (
    check_consensus,
    process_controversial_clusters,
    interactive_consensus_annotation,
    print_consensus_summary,
    facilitate_cluster_discussion,
    summarize_discussion
)
from .compare import (
    compare_model_predictions,
    create_comparison_table,
    analyze_confusion_patterns
)

__version__ = '0.1.0'

__all__ = [
    # Core annotation
    'annotate_clusters',
    'batch_annotate_clusters',
    'get_model_response',

    # Functions
    'get_provider',
    'clean_annotation',
    'identify_controversial_clusters',
    'select_best_prediction',

    # Logging
    'setup_logging',
    'write_log',

    # Utils
    'load_api_key',
    'create_cache_key',
    'save_to_cache',
    'load_from_cache',
    'validate_cache',
    'clear_cache',
    'get_cache_stats',
    'format_results',
    'find_agreement',

    # Prompts
    'create_prompt',
    'create_batch_prompt',
    'create_json_prompt',
    'create_discussion_prompt',
    'create_consensus_check_prompt',
    'create_initial_discussion_prompt',

    # Consensus
    'check_consensus',
    'process_controversial_clusters',
    'interactive_consensus_annotation',
    'print_consensus_summary',
    'facilitate_cluster_discussion',
    'summarize_discussion',

    # Compare
    'compare_model_predictions',
    'create_comparison_table',
    'analyze_confusion_patterns'
]
