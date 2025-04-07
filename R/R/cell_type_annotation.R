# Define global variables
utils::globalVariables(c("cluster", "avg_log2FC", "gene"))

#' @importFrom dplyr group_by top_n group_split slice_head pull
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
#' @param input One of the following:
#'   - A data frame from Seurat's FindAllMarkers() function containing differential gene expression results
#'   - A list where each element has a 'genes' field containing marker genes for a cluster
#' @param tissue_name Character string specifying the tissue type or cell source (e.g., 'human PBMC', 
#'   'mouse brain'). This helps provide context for more accurate annotations.
#' @param model Character string specifying the LLM model to use. Supported models:
#'   - OpenAI: 'gpt-4o', 'o1' 
#'   - Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
#'   - DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
#'   - Google: 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
#'   - Alibaba: 'qwen-max-2025-01-25'
#'   - Stepfun: 'step-2-16k', 'step-2-mini', 'step-1-8k'
#'   - Zhipu: 'glm-4-plus', 'glm-3-turbo'
#'   - MiniMax: 'minimax-text-01'
#' @param api_key Character string containing the API key for the selected model. 
#'   If NA, returns the generated prompt without making an API call.
#' @param top_gene_count Integer specifying the number of top marker genes to use per cluster.
#' @param debug Logical. If TRUE, prints additional debugging information during execution.
#'   when input is from Seurat's FindAllMarkers(). Default: 10

#' @import httr
#' @import jsonlite
#' @export
#' 
#' @return A character vector containing:
#'   - When api_key is provided: One cell type annotation per cluster, in the order of input clusters
#'   - When api_key is NA: The generated prompt string that would be sent to the LLM
#'
#' @examples 
#' # Example 1: Using custom gene lists, returning prompt only
#' annotate_cell_types(
#'   input = list(
#'     t_cells = list(genes = c('CD3D', 'CD3E', 'CD3G', 'CD28')),
#'     b_cells = list(genes = c('CD19', 'CD79A', 'CD79B', 'MS4A1')),
#'     monocytes = list(genes = c('CD14', 'CD68', 'CSF1R', 'FCGR3A'))
#'   ),
#'   tissue_name = 'human PBMC',
#'   model = 'gpt-4o'
#' )
#'
#' # Example 2: Using with Seurat pipeline
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
#' # Get cell type annotations
#' cell_types <- annotate_cell_types(
#'   input = all.markers,
#'   tissue_name = 'human PBMC',
#'   model = 'gpt-4o',
#'   api_key = 'your-api-key',
#'   top_gene_count = 15
#' )
#' }
#'
#' @seealso 
#' * [Seurat::FindAllMarkers()]
#' * [mLLMCelltype::get_provider()]
#' * [mLLMCelltype::process_openai()]

annotate_cell_types <- function(input, 
                               tissue_name = NULL, 
                               model = 'gpt-4o', 
                               api_key = NA,
                               top_gene_count = 10,
                               debug = FALSE) {
  
  # Check if tissue_name is provided
  if (is.null(tissue_name)) {
    stop("tissue_name parameter is required. Please specify the tissue type or cell source (e.g., 'human PBMC', 'mouse brain').")
  }
  
  # Determine provider from model name
  provider <- get_provider(model)
  
  # Log model and provider information
  write_log(sprintf("Processing input with Model: %s (Provider: %s)", model, provider))
  
  # Generate prompt using the dedicated function
  prompt_result <- create_annotation_prompt(input, tissue_name, top_gene_count)
  prompt <- prompt_result$prompt
  
  # If debug mode is enabled, print more information
  if (debug) {
    cat("\n==== DEBUG INFO ====\n")
    cat("Gene lists structure:\n")
    utils::str(prompt_result$gene_lists)
    cat("\nFormatted prompt (raw):\n")
    cat(prompt, "\n")
    cat("==== END DEBUG INFO ====\n\n")
  }
  
  # Log gene lists
  write_log("\nGene lists for each cluster:")
  cluster_ids <- names(prompt_result$gene_lists)
  for (id in cluster_ids) {
    write_log(sprintf("Cluster %s: %s", id, prompt_result$gene_lists[[id]]))
  }
  
  write_log("\nGenerated prompt:")
  write_log(prompt)
  
  # If no API key, return prompt
  if (is.na(api_key)) {
    return(prompt)
  }
  
  # Process based on provider
  result <- switch(provider,
    "openai" = process_openai(prompt, model, api_key),
    "anthropic" = process_anthropic(prompt, model, api_key),
    "deepseek" = process_deepseek(prompt, model, api_key),
    "gemini" = process_gemini(prompt, model, api_key),
    "qwen" = process_qwen(prompt, model, api_key),
    "stepfun" = process_stepfun(prompt, model, api_key),
    "zhipu" = process_zhipu(prompt, model, api_key),
    "minimax" = process_minimax(prompt, model, api_key)
  )
  
  write_log("\nModel response:")
  write_log(result)
  
  return(result)
}
