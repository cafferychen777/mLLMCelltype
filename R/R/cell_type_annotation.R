#' @keywords internal
"_PACKAGE"

#' @importFrom dplyr group_by top_n group_split
#' @importFrom utils head

#' @title Cell Type Annotation with Multi-LLM Framework
#' @name annotate_cell_types
#'
#' @description
#' A comprehensive function for automated cell type annotation using multiple Large Language Models (LLMs).
#' This function supports both Seurat's differential gene expression results and custom gene lists as input.
#' It implements a sophisticated annotation pipeline that leverages state-of-the-art LLMs to identify
#' cell types based on marker gene expression patterns.
#'
#
#'   - A data frame from Seurat's FindAllMarkers() function containing differential gene expression results
#'     (must have columns: 'cluster', 'gene', and 'avg_log2FC'). The function will select the top genes
#'     based on avg_log2FC for each cluster.
#'   - A list where each element has a 'genes' field containing marker genes for a cluster.
#'     This can be in one of these formats:
#'     * Named with cluster IDs: list("0" = list(genes = c(...)), "1" = list(genes = c(...)))
#'     * Named with cell type names: list(t_cells = list(genes = c(...)), b_cells = list(genes = c(...)))
#'     * Unnamed list: list(list(genes = c(...)), list(genes = c(...)))
#'   - For both input types, if cluster IDs are numeric and start from 1, they will be automatically
#'     converted to 0-based indexing (e.g., cluster 1 becomes cluster 0) for consistency.
#'
#'   IMPORTANT NOTE ON CLUSTER IDs:
#'   The 'cluster' column must contain numeric values or values that can be converted to numeric.
#'   Non-numeric cluster IDs (e.g., "cluster_1", "T_cells", "7_0") may cause errors or unexpected
#'   behavior. Before using this function, it is recommended to:
#'
#'   1. Ensure all cluster IDs are numeric or can be cleanly converted to numeric values
#'   2. If your data contains non-numeric cluster IDs, consider creating a mapping between
#'      original IDs and numeric IDs:
#'      ```r
#'      # Example of standardizing cluster IDs
#'      original_ids <- unique(markers$cluster)
#'      id_mapping <- data.frame(
#'        original = original_ids,
#'        numeric = seq(0, length(original_ids) - 1)
#'      )
#'      markers$cluster <- id_mapping$numeric[match(markers$cluster, id_mapping$original)]
#'      ```
#
#'   'mouse brain'). This helps provide context for more accurate annotations.
#
#'   - OpenAI: 'gpt-5.2', 'gpt-5.1', 'gpt-5', 'gpt-4.1', 'gpt-4o', 'o3-pro', 'o3', 'o4-mini', 'o1', 'o1-pro'
#'   - Anthropic: 'claude-opus-4-6-20260205', 'claude-opus-4-5-20251101', 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001',
#'     'claude-opus-4-1-20250805', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219'
#'   - DeepSeek: 'deepseek-chat', 'deepseek-reasoner', 'deepseek-r1'
#'   - Google: 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash'
#'   - Alibaba: 'qwen3-max', 'qwen-max-2025-01-25', 'qwen-plus'
#'   - Stepfun: 'step-3', 'step-2-16k', 'step-2-mini'
#'   - Zhipu: 'glm-4.7', 'glm-4-plus'
#'   - MiniMax: 'minimax-m2.1', 'minimax-m2', 'MiniMax-Text-01'
#'   - X.AI: 'grok-4', 'grok-4.1', 'grok-4-heavy', 'grok-3', 'grok-3-fast', 'grok-3-mini'
#'   - OpenRouter: Provides access to models from multiple providers through a single API. Format: 'provider/model-name'
#'     - OpenAI models: 'openai/gpt-5.2', 'openai/gpt-5', 'openai/o3-pro', 'openai/o4-mini'
#'     - Anthropic models: 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5'
#'     - Meta models: 'meta-llama/llama-4-maverick', 'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct'
#'     - Google models: 'google/gemini-3-pro', 'google/gemini-3-flash', 'google/gemini-2.5-pro'
#'     - Mistral models: 'mistralai/mistral-large', 'mistralai/magistral-medium-2506'
#'     - Other models: 'deepseek/deepseek-r1', 'deepseek/deepseek-chat-v3.1', 'microsoft/mai-ds-r1'
#
#'   Each provider requires a specific API key format and authentication method:
#'
#'   - OpenAI: "sk-..." (obtain from OpenAI platform)
#'   - Anthropic: "sk-ant-..." (obtain from Anthropic console)
#'   - Google: A Google API key for Gemini models (obtain from Google AI)
#'   - DeepSeek: API key from DeepSeek platform
#'   - Qwen: API key from Alibaba Cloud
#'   - Stepfun: API key from Stepfun AI
#'   - Zhipu: API key from Zhipu AI
#'   - MiniMax: API key from MiniMax
#'   - X.AI: API key for Grok models
#'   - OpenRouter: "sk-or-..." (obtain from OpenRouter)
#'     OpenRouter provides access to multiple models through a single API key
#'
#'   The API key can be provided directly or stored in environment variables:
#'   ```r
#'   # Direct API key
#'   result <- annotate_cell_types(input, tissue_name, model="gpt-5.2",
#'                                api_key="sk-...")
#'
#'   # Using environment variables
#'   Sys.setenv(OPENAI_API_KEY="sk-...")
#'   Sys.setenv(ANTHROPIC_API_KEY="sk-ant-...")
#'   Sys.setenv(OPENROUTER_API_KEY="sk-or-...")
#'
#'   # Then use with environment variables
#'   result <- annotate_cell_types(input, tissue_name, model="claude-sonnet-4-5-20250929",
#'                                api_key=Sys.getenv("ANTHROPIC_API_KEY"))
#'   ```
#'
#'   If NA, returns the generated prompt without making an API call, which is useful for
#'   reviewing the prompt before sending it to the API.
#
#'   when input is from Seurat's FindAllMarkers(). Default: 10
#
#
#'   - A single character string: Applied to all providers (e.g., "https://api.proxy.com/v1")
#'   - A named list: Provider-specific URLs (e.g., list(openai = "https://openai-proxy.com/v1",
#'     anthropic = "https://anthropic-proxy.com/v1")). This is useful for:
#'     * Users accessing international APIs through proxies
#'     * Enterprise users with internal API gateways
#'     * Development/testing with local or alternative endpoints
#'   If NULL (default), uses official API endpoints for each provider.
#'
#' @param input Either a data frame from Seurat's FindAllMarkers() containing columns 'cluster', 'gene', and 'avg_log2FC', or a list with 'genes' field for each cluster
#' @param tissue_name Optional tissue context (e.g., 'human PBMC', 'mouse brain') for more accurate annotations
#' @param model Model name to use. Default: 'gpt-5.2'. See details for supported models
#' @param api_key API key for the selected model provider. If NA, returns prompt only
#' @param top_gene_count Number of top genes to use per cluster when input is from Seurat. Default: 10
#' @param debug Logical indicating whether to enable debug output. Default: FALSE
#' @param base_urls Optional base URLs for API endpoints. Can be a string or named list for custom endpoints
#'
#' @return When api_key is provided: Vector of cell type annotations per cluster. When api_key is NA: The generated prompt string
#'
#' @importFrom httr POST add_headers content http_error status_code timeout
#' @importFrom jsonlite toJSON
#' @examples
#' # Example 1: Using custom gene lists, returning prompt only (no API call)
#' annotate_cell_types(
#'   input = list(
#'     t_cells = list(genes = c('CD3D', 'CD3E', 'CD3G', 'CD28')),
#'     b_cells = list(genes = c('CD19', 'CD79A', 'CD79B', 'MS4A1')),
#'     monocytes = list(genes = c('CD14', 'CD68', 'CSF1R', 'FCGR3A'))
#'   ),
#'   tissue_name = 'human PBMC',
#'   model = 'gpt-5.2',
#'   api_key = NA  # Returns prompt only without making API call
#' )
#'
#' # Example 2: Using with Seurat pipeline and OpenAI model
#' \dontrun{
#' library(Seurat)
#'
#' # Load example data
#' data("pbmc_small")
#'
#' # Find marker genes
#' all.markers <- FindAllMarkers(
#'   object = pbmc_small,
#'   only.pos = TRUE,
#'   min.pct = 0.25,
#'   logfc.threshold = 0.25
#' )
#'
#' # Set API key in environment variable (recommended approach)
#' Sys.setenv(OPENAI_API_KEY = "your-openai-api-key")
#'
#' # Get cell type annotations using OpenAI model
#' openai_annotations <- annotate_cell_types(
#'   input = all.markers,
#'   tissue_name = 'human PBMC',
#'   model = 'gpt-5.2',
#'   api_key = Sys.getenv("OPENAI_API_KEY"),
#'   top_gene_count = 15
#' )
#'
#' # Example 3: Using Anthropic Claude model
#' Sys.setenv(ANTHROPIC_API_KEY = "your-anthropic-api-key")
#'
#' claude_annotations <- annotate_cell_types(
#'   input = all.markers,
#'   tissue_name = 'human PBMC',
#'   model = 'claude-opus-4-6-20260205',
#'   api_key = Sys.getenv("ANTHROPIC_API_KEY"),
#'   top_gene_count = 15
#' )
#'
#' # Example 4: Using OpenRouter to access multiple models
#' Sys.setenv(OPENROUTER_API_KEY = "your-openrouter-api-key")
#'
#' # Access OpenAI models through OpenRouter
#' openrouter_gpt4_annotations <- annotate_cell_types(
#'   input = all.markers,
#'   tissue_name = 'human PBMC',
#'   model = 'openai/gpt-5.2',  # Note the provider/model format
#'   api_key = Sys.getenv("OPENROUTER_API_KEY"),
#'   top_gene_count = 15
#' )
#'
#' # Access Anthropic models through OpenRouter
#' openrouter_claude_annotations <- annotate_cell_types(
#'   input = all.markers,
#'   tissue_name = 'human PBMC',
#'   model = 'anthropic/claude-opus-4.6',  # Note the provider/model format
#'   api_key = Sys.getenv("OPENROUTER_API_KEY"),
#'   top_gene_count = 15
#' )
#'
#' # Example 5: Using with mouse brain data
#' mouse_annotations <- annotate_cell_types(
#'   input = mouse_markers,  # Your mouse marker genes
#'   tissue_name = 'mouse brain',  # Specify correct tissue for context
#'   model = 'gpt-5.2',
#'   api_key = Sys.getenv("OPENAI_API_KEY"),
#'   top_gene_count = 20,  # Use more genes for complex tissues
#'   debug = TRUE  # Enable debug output
#' )
#' }
#'
#' @seealso
#' * [Seurat::FindAllMarkers()]
#' * [mLLMCelltype::get_provider()]
#' * [mLLMCelltype::process_openai()]
#' @export
annotate_cell_types <- function(input,
                               tissue_name = NULL,
                               model = 'gpt-5.2',
                               api_key = NA,
                               top_gene_count = 10,
                               debug = FALSE,
                               base_urls = NULL) {

  # Check if tissue_name is provided
  if (is.null(tissue_name)) {
    stop("tissue_name parameter is required. Please specify the tissue type or cell source (e.g., 'human PBMC', 'mouse brain').")
  }

  # Determine provider from model name
  provider <- get_provider(model)

  # Resolve provider-specific base URL
  provider_base_url <- resolve_provider_base_url(provider, base_urls)

  # Log model and provider information
  log_info("Processing input with model and provider", list(model = model, provider = provider, custom_url = !is.null(provider_base_url)))

  # Generate prompt using the dedicated function
  prompt_result <- create_annotation_prompt(input, tissue_name, top_gene_count)
  prompt <- prompt_result$prompt

  # If debug mode is enabled, print more information
  if (debug) {
    message("\n==== DEBUG INFO ====\n")
    message("Gene lists structure:\n")
    utils::str(prompt_result$gene_lists)
    message("\nFormatted prompt (raw):\n")
    message(prompt)
    message("==== END DEBUG INFO ====\n\n")
  }

  # Log gene lists
  log_debug("Gene lists for each cluster")
  cluster_ids <- names(prompt_result$gene_lists)
  for (id in cluster_ids) {
    log_debug("Cluster gene list", list(cluster = id, genes = prompt_result$gene_lists[[id]]))
  }

  log_debug("Generated prompt", list(prompt = prompt))

  # If no API key, return prompt
  if (is.na(api_key)) {
    return(prompt)
  }

  # Check for custom provider first (consistent with get_model_response.R)
  if (exists(provider, envir = custom_providers)) {
    log_debug("Using custom provider", list(provider = provider))
    result <- process_custom(prompt, model, api_key)
  } else {
    # Process based on built-in provider
    result <- switch(provider,
      "openai" = process_openai(prompt, model, api_key, provider_base_url),
      "anthropic" = process_anthropic(prompt, model, api_key, provider_base_url),
      "deepseek" = process_deepseek(prompt, model, api_key, provider_base_url),
      "gemini" = process_gemini(prompt, model, api_key, provider_base_url),
      "qwen" = process_qwen(prompt, model, api_key, provider_base_url),
      "stepfun" = process_stepfun(prompt, model, api_key, provider_base_url),
      "zhipu" = process_zhipu(prompt, model, api_key, provider_base_url),
      "minimax" = process_minimax(prompt, model, api_key, provider_base_url),
      "grok" = process_grok(prompt, model, api_key, provider_base_url),
      "openrouter" = process_openrouter(prompt, model, api_key, provider_base_url),
      stop("Unsupported model provider: ", provider)
    )
  }

  log_info("Model response received", list(response = result))

  return(result)
}
