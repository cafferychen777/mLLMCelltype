# Interactive consensus building for cell type annotation

This function implements an interactive voting and discussion mechanism
where multiple LLMs collaborate to reach a consensus on cell type
annotations, particularly focusing on clusters with low agreement. The
process includes:

1.  Initial voting by all LLMs

2.  Identification of controversial clusters

3.  Detailed discussion for controversial clusters

4.  Final summary by a designated LLM (default: Claude)

## Usage

``` r
interactive_consensus_annotation(
  input,
  tissue_name = NULL,
  models = c("claude-opus-4-6-20260205", "gpt-5.2", "gemini-3-pro", "deepseek-r1",
    "grok-4.1"),
  api_keys,
  top_gene_count = 10,
  controversy_threshold = 0.7,
  entropy_threshold = 1,
  max_discussion_rounds = 3,
  consensus_check_model = NULL,
  log_dir = "logs",
  cache_dir = NULL,
  use_cache = TRUE,
  base_urls = NULL,
  clusters_to_analyze = NULL,
  force_rerun = FALSE
)
```

## Arguments

- input:

  Either a data frame from Seurat's FindAllMarkers() function containing
  differential gene expression results (must have columns: 'cluster',
  'gene', and 'avg_log2FC'), or a list where each element is either a
  character vector of genes or a list containing a `genes` field.

- tissue_name:

  Character string specifying the tissue type for context-aware cell
  type annotation. If NULL, generic cell type annotation will be
  performed.

- models:

  Character vector of model names to use for consensus annotation.
  Minimum 2 models required. Supports models from OpenAI, Anthropic,
  DeepSeek, Google, Alibaba, Stepfun, Zhipu, MiniMax, X.AI, and
  OpenRouter.

- api_keys:

  Named list of API keys. Can use provider names as keys (e.g.,
  "openai", "anthropic") or model names as keys (e.g., "gpt-5").

- top_gene_count:

  Integer specifying the number of top marker genes to use for
  annotation per cluster (default: 10).

- controversy_threshold:

  Numeric value between 0 and 1 for consensus proportion threshold.
  Clusters below this threshold are considered controversial (default:
  0.7).

- entropy_threshold:

  Numeric value for entropy threshold. Higher entropy indicates more
  disagreement among models (default: 1.0).

- max_discussion_rounds:

  Integer specifying maximum number of discussion rounds for
  controversial clusters (default: 3).

- consensus_check_model:

  Character string specifying which model to use for consensus checking.
  If NULL, uses the first model from the models list.

- log_dir:

  Character scalar specifying directory for log files (default: "logs").
  This function reinitializes the session logger with this directory at
  the start of each call.

- cache_dir:

  Character string or NULL. Cache directory for storing results. NULL
  uses system cache, "local" uses current directory, "temp" uses
  temporary directory, or specify custom path.

- use_cache:

  Logical indicating whether to use caching (default: TRUE).

- base_urls:

  Named list or character string specifying custom API base URLs. Useful
  for proxies or alternative endpoints. If NULL, uses official
  endpoints.

- clusters_to_analyze:

  Character or numeric vector specifying which clusters to analyze. If
  NULL (default), all clusters are analyzed.

- force_rerun:

  Logical indicating whether to force rerun of all specified clusters,
  ignoring cache. Only affects controversial cluster discussions
  (default: FALSE).

## Value

A list containing:

- `initial_results`: Initial voting results, consensus checks, and
  controversial cluster IDs

- `final_annotations`: Final annotations keyed by cluster ID

- `controversial_clusters`: Clusters identified as controversial

- `discussion_logs`: Detailed discussion logs for controversial clusters

- `session_id`: Logger session identifier

- `voting_results`: Backward-compatible alias of `initial_results`

- `discussion_results`: Backward-compatible alias of `discussion_logs`

- `final_consensus`: Backward-compatible alias of `final_annotations`
