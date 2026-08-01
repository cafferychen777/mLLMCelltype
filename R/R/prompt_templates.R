# Prompt templates for mLLMCelltype
#
# This file contains all prompt template functions used in mLLMCelltype.
# These functions create various prompts for different stages of the cell type annotation process.

.normalize_top_gene_count <- function(top_gene_count) {
  .normalize_positive_integer(top_gene_count, "top_gene_count")
}

.normalize_marker_gene_vector <- function(genes) {
  if (is.list(genes) || is.data.frame(genes)) {
    stop("marker genes must be an atomic vector")
  }
  normalized <- trimws(as.character(genes))
  unique(normalized[!is.na(normalized) & nzchar(normalized)])
}

extract_cluster_item_genes <- function(item) {
  if (is.list(item) && "genes" %in% names(item)) {
    return(.normalize_marker_gene_vector(item$genes))
  }
  if (is.atomic(item)) {
    return(.normalize_marker_gene_vector(item))
  }
  stop("When input is a list, each element must be a character vector of genes (or another atomic gene vector) or a list containing a 'genes' field")
}

select_cluster_marker_genes <- function(input, cluster_id, top_gene_count) {
  cluster_key <- trimws(as.character(cluster_id))
  top_gene_count <- .normalize_top_gene_count(top_gene_count)

  if (is.list(input) && !is.data.frame(input)) {
    cluster_names <- names(input)
    if (is.null(cluster_names)) {
      cluster_names <- as.character(seq_along(input) - 1)
    } else {
      cluster_names <- trimws(cluster_names)
    }
    if (anyNA(cluster_names) || any(!nzchar(cluster_names))) {
      stop("Cluster IDs must be non-empty strings")
    }
    if (anyDuplicated(cluster_names)) {
      stop("Duplicate cluster IDs detected after normalization")
    }

    cluster_index <- match(cluster_key, cluster_names)
    if (is.na(cluster_index)) {
      stop(sprintf("Cluster '%s' was not found in list input", cluster_key))
    }
    genes <- head(extract_cluster_item_genes(input[[cluster_index]]), top_gene_count)
  } else if (is.data.frame(input)) {
    required_columns <- c("cluster", "gene")
    if (!all(required_columns %in% names(input))) {
      stop("Data frame input must contain 'cluster' and 'gene' columns")
    }

    normalized_clusters <- trimws(as.character(input$cluster))
    cluster_rows <- input[
      !is.na(normalized_clusters) & normalized_clusters == cluster_key,
      ,
      drop = FALSE
    ]
    if ("avg_log2FC" %in% names(cluster_rows)) {
      rank_order <- order(cluster_rows$avg_log2FC, decreasing = TRUE, na.last = NA)
      cluster_rows <- cluster_rows[rank_order, , drop = FALSE]
    }
    genes <- head(.normalize_marker_gene_vector(cluster_rows$gene), top_gene_count)
  } else {
    stop("Input must be either a data.frame or a list of gene lists")
  }

  if (length(genes) == 0) {
    stop(sprintf("No genes found for cluster '%s'", cluster_key))
  }
  genes
}

#' Normalize list input into a canonical cluster->genes mapping
#'
#' For list input, each element can be either:
#' 1) a list containing a `genes` field, or
#' 2) a character vector of genes.
#'
#' Naming rules:
#' - unnamed lists are assigned 0-based IDs ("0", "1", ...)
#' - numeric names are preserved as-is (e.g., "1", "2", "3" stays unchanged)
#' - non-numeric names are preserved as-is
#'
#' @param input List input for cluster annotation
#' @return Named list of character vectors (cluster_id -> genes)
#' @keywords internal
normalize_cluster_gene_list <- function(input) {
  if (!is.list(input) || is.data.frame(input)) {
    stop("normalize_cluster_gene_list expects a list input")
  }
  if (length(input) == 0) {
    stop("input must contain at least one cluster")
  }

  names_vec <- names(input)
  gene_vectors <- lapply(input, extract_cluster_item_genes)

  if (is.null(names_vec)) {
    canonical_names <- as.character(seq_along(gene_vectors) - 1)
  } else {
    canonical_names <- trimws(names_vec)
    if (any(is.na(canonical_names)) || any(!nzchar(canonical_names))) {
      stop("Cluster IDs must be non-empty strings")
    }
  }

  if (anyDuplicated(canonical_names)) {
    stop("Duplicate cluster IDs detected after normalization")
  }

  names(gene_vectors) <- canonical_names
  empty_clusters <- canonical_names[lengths(gene_vectors) == 0]
  if (length(empty_clusters) > 0) {
    stop(
      "No genes found for cluster(s): ",
      paste(empty_clusters, collapse = ", ")
    )
  }
  gene_vectors
}

canonical_cluster_ids <- function(input) {
  cluster_ids <- if (is.list(input) && !is.data.frame(input)) {
    input_names <- names(input)
    if (is.null(input_names)) {
      as.character(seq_along(input) - 1)
    } else {
      trimws(input_names)
    }
  } else if (is.data.frame(input) && "cluster" %in% names(input)) {
    normalized <- trimws(as.character(input$cluster))
    unique(normalized[!is.na(normalized) & nzchar(normalized)])
  } else {
    stop("Input must provide cluster IDs through list names or a cluster column")
  }

  if (anyNA(cluster_ids) || any(!nzchar(cluster_ids))) {
    stop("Cluster IDs must be non-empty strings")
  }
  if (anyDuplicated(cluster_ids)) {
    stop("Duplicate cluster IDs detected after normalization")
  }
  if (length(cluster_ids) > 0 &&
      all(!is.na(suppressWarnings(as.numeric(cluster_ids))))) {
    cluster_ids <- cluster_ids[order(as.numeric(cluster_ids))]
  }
  cluster_ids
}

#' Extract marker genes for a discussion prompt
#'
#' @param input Marker gene input in list or data.frame format
#' @param cluster_id Cluster ID to extract
#' @param top_gene_count Maximum number of genes to include
#' @return Comma-separated marker genes for the requested cluster
#' @keywords internal
#' @noRd
extract_cluster_genes_for_discussion <- function(input, cluster_id, top_gene_count) {
  paste(
    select_cluster_marker_genes(input, cluster_id, top_gene_count),
    collapse = ","
  )
}

#' Prepare normalized data for annotation prompts
#'
#' This is the single input-normalization path shared by the standard and
#' reasoning annotation prompts. Prompt renderers consume the returned data
#' without duplicating validation, marker selection, or cluster ordering.
#'
#' @param input Either a data frame from Seurat's FindAllMarkers() or a list for each cluster
#'   where each element is either a character vector of genes or a list containing a `genes` field
#'   Cluster IDs in named inputs are preserved as-is; unnamed list input receives sequential IDs starting at "0".
#' @param tissue_name Tissue context for the annotation (e.g., 'human PBMC', 'mouse brain')
#' @param top_gene_count Number of top genes to use per cluster when input is from Seurat
#'
#' @return A list with `tissue_name` (normalized), `marker_text` (one
#'   "cluster_id: gene1, gene2" line per cluster, in canonical order),
#'   `expected_count` (number of clusters), and `gene_lists` (cluster ID to
#'   marker genes mapping).
#' @keywords internal
#' @noRd
prepare_annotation_prompt_data <- function(input, tissue_name, top_gene_count) {
  tissue_name <- .normalize_required_string(tissue_name, "tissue_name")
  top_gene_count <- .normalize_top_gene_count(top_gene_count)

  if (is.list(input) && !is.data.frame(input)) {
    normalized_input <- normalize_cluster_gene_list(input)

    gene_lists <- list()

    for (cluster_id in names(normalized_input)) {
      genes <- normalized_input[[cluster_id]]
      gene_lists[[cluster_id]] <- paste(genes, collapse = ", ")
    }

    expected_count <- length(normalized_input)
  } else if (is.data.frame(input)) {
    required_columns <- c("cluster", "gene", "avg_log2FC")
    column_counts <- vapply(required_columns, function(column) {
      sum(names(input) == column, na.rm = TRUE)
    }, integer(1))
    if (any(column_counts != 1)) {
      stop(
        "Data frame input must contain exactly one each of: ",
        paste(required_columns, collapse = ", ")
      )
    }
    if (!is.numeric(input$avg_log2FC)) {
      stop("avg_log2FC must be numeric")
    }

    normalized_clusters <- trimws(as.character(input$cluster))
    valid_cluster_rows <- !is.na(normalized_clusters) & nzchar(normalized_clusters)
    if (any(!valid_cluster_rows)) {
      warning("Skipping rows with missing or empty cluster IDs", call. = FALSE)
    }
    input <- input[valid_cluster_rows, , drop = FALSE]
    normalized_clusters <- normalized_clusters[valid_cluster_rows]
    if (nrow(input) == 0) {
      stop("input must contain at least one valid cluster")
    }

    cluster_names <- unique(normalized_clusters)
    gene_lists <- list()
    for (cluster_id in cluster_names) {
      genes <- select_cluster_marker_genes(input, cluster_id, top_gene_count)
      gene_lists[[cluster_id]] <- paste(genes, collapse = ",")
    }

    expected_count <- length(gene_lists)
  } else {
    stop("Input must be either a data.frame (from Seurat) or a list of gene lists")
  }

  cluster_names <- canonical_cluster_ids(gene_lists)
  gene_lists <- gene_lists[cluster_names]

  # Create formatted lines from gene_lists using the canonical cluster order.
  formatted_lines <- vapply(cluster_names, function(name) {
    paste0(name, ": ", gene_lists[[name]])
  }, character(1), USE.NAMES = FALSE)

  return(list(
    tissue_name = tissue_name,
    marker_text = paste(formatted_lines, collapse = "\n"),
    expected_count = expected_count,
    gene_lists = gene_lists
  ))
}

#' Create prompt for cell type annotation
#'
#' @param input Either a data frame from Seurat's FindAllMarkers() or a list for each cluster
#'   where each element is either a character vector of genes or a list containing a `genes` field
#'   Cluster IDs in named inputs are preserved as-is; unnamed list input receives sequential IDs starting at "0".
#' @param tissue_name Tissue context for the annotation (e.g., 'human PBMC', 'mouse brain')
#' @param top_gene_count Number of top genes to use per cluster when input is from Seurat. Default: 10
#'
#' @return A list with `prompt` (formatted prompt text), `expected_count`
#'   (number of clusters), and `gene_lists` (cluster ID to marker genes mapping).
#' @export
create_annotation_prompt <- function(input, tissue_name, top_gene_count = 10) {
  prompt_data <- prepare_annotation_prompt_data(input, tissue_name, top_gene_count)

  prompt <- paste0("You are a cell type annotation expert. Below are marker genes for different cell clusters in ", 
                  prompt_data$tissue_name, ".\n\n",
                  prompt_data$marker_text,
                  "\n\nReturn exactly one cell type name per line, in the same order as the clusters shown above, without cluster IDs or explanation.")
  
  return(list(
    prompt = prompt,
    expected_count = prompt_data$expected_count,
    gene_lists = prompt_data$gene_lists
  ))
}

#' Create reasoning-aware prompt for cell type annotation
#'
#' This prompt is built from its own template rather than derived from
#' create_annotation_prompt(), so the two prompts can evolve independently.
#' The template mirrors REASONING_PROMPT_TEMPLATE in the Python package
#' (python/mllmcelltype/prompts.py) so both implementations ask for the same
#' structured JSON output; it is adapted to the R signature, which takes a
#' single `tissue_name` instead of separate species/tissue arguments.
#'
#' @param input Either a data frame from Seurat's FindAllMarkers() or a list for each cluster
#'   where each element is either a character vector of genes or a list containing a `genes` field.
#' @param tissue_name Tissue context for the annotation (e.g., 'human PBMC', 'mouse brain')
#' @param top_gene_count Number of top genes to use per cluster when input is from Seurat. Default: 10
#'
#' @return A list with `prompt` (formatted prompt text), `expected_count`
#'   (number of clusters), and `gene_lists` (cluster ID to marker genes mapping).
#' @export
create_reasoning_annotation_prompt <- function(input, tissue_name, top_gene_count = 10) {
  prompt_data <- prepare_annotation_prompt_data(input, tissue_name, top_gene_count)

  reasoning_prompt <- paste0(
    "You are an expert single-cell RNA-seq analyst specializing in cell type annotation.\n",
    "I need you to identify cell types in ", prompt_data$tissue_name, ".\n",
    "Below is a list of marker genes for each cluster.\n",
    "Please assign the most likely cell type to each cluster based on the marker genes.\n\n",
    "For each cluster, return a JSON object with the following fields:\n",
    "- \"cluster_id\": the cluster ID as shown below\n",
    "- \"cell_type\": the annotated cell type\n",
    "- \"marker_genes\": the marker genes from the provided gene set that support this annotation, ",
    "comma-separated, preserving the original case exactly as shown below\n",
    "- \"gene_expression\": a concise natural-language description of where each provided ",
    "differential gene is typically expressed across cell types\n\n",
    "Return a single JSON array containing one object per cluster, in the same order as the clusters shown below. ",
    "Do not include markdown formatting or any explanation outside the JSON array.\n\n",
    "Here are the marker genes for each cluster:\n",
    prompt_data$marker_text
  )

  return(list(
    prompt = reasoning_prompt,
    expected_count = prompt_data$expected_count,
    gene_lists = prompt_data$gene_lists
  ))
}

#' Create prompt for checking consensus among model predictions
#
#
#
#
#' @keywords internal
create_consensus_check_prompt <- function(round_responses, controversy_threshold = 2/3, entropy_threshold = 1.0) {
  # Format the predictions for Claude
  paste(
    "You are a cell type annotation expert. Below are different models' predictions for the same cell cluster.",
    "Your task is to analyze these predictions and calculate uncertainty metrics.",
    "",
    "PREDICTIONS:",
    paste(paste("Model", seq_along(round_responses), ":", round_responses), collapse = "\n"),
    "",
    "IMPORTANT GUIDELINES:",
    "1. Consider predictions as matching only when they name the same biological entity, allowing:",
    "   - Synonymous nomenclature (e.g., 'NK cells' vs 'Natural Killer cells')",
    "   - Formatting and capitalization differences",
    "2. Preserve biologically meaningful subtype or state qualifiers such as activated, mature, memory, regulatory, CD4+, and CD8+; do not merge labels when doing so would lose specificity",
    "3. Group only predictions that remain biologically equivalent at the same level of granularity",
    "4. If any prediction is 'Unknown' or 'Unclear', treat it as a separate group",
    "",
    "CALCULATE THE FOLLOWING METRICS:",
    "1. Consensus Proportion = Number of models supporting the majority prediction / Total number of models",
    "2. Shannon Entropy = -sum(p_i * log2(p_i)) where p_i is the proportion of models predicting each unique cell type",
    sprintf("3. Determine if consensus is reached (Consensus Proportion >= %s AND Entropy <= %s)",
            format(controversy_threshold, nsmall=1), format(entropy_threshold, nsmall=1)),
    "",
    "RESPONSE FORMAT:",
    "Line 1: 1 if consensus is reached, 0 if not",
    "Line 2: Consensus Proportion (a decimal between 0 and 1)",
    "Line 3: Shannon Entropy (a decimal number)",
    "Line 4: The majority cell type prediction",
    "",
    "Example matches:",
    "- 'NK cells' = 'Natural Killer cells'",
    "- 'CD8+ T cells' = 'Cytotoxic T cells'",
    "- 'B cells' = 'B lymphocytes'",
    "",
    "RESPOND WITH EXACTLY FOUR LINES AS SPECIFIED ABOVE.",
    sep = "\n"
  )
}

format_named_model_responses <- function(responses,
                                         separator = ":\n",
                                         missing = "No prediction",
                                         validator = is_valid_model_response) {
  response_names <- names(responses)
  if (!is.list(responses) || is.null(response_names) ||
      anyNA(response_names) || any(!nzchar(trimws(response_names)))) {
    stop("responses must be a named list with non-empty model names")
  }

  vapply(seq_along(responses), function(index) {
    response <- responses[[index]]
    response_text <- if (isTRUE(validator(response))) {
      paste(response, collapse = "\n")
    } else {
      missing
    }
    paste0(trimws(response_names[[index]]), separator, response_text)
  }, character(1), USE.NAMES = FALSE)
}

#' Create prompt for additional discussion rounds
#
#
#
#
#
#
#' @keywords internal
create_discussion_prompt <- function(cluster_id,
                                     cluster_genes,
                                     tissue_name,
                                     previous_rounds,
                                     round_number) {
  # Compile previous discussion history
  discussion_history <- vapply(previous_rounds, function(round) {
    responses <- round$responses
    sprintf(
      "Round %d:\n%s",
      round$round_number,
      paste(format_named_model_responses(responses), collapse = "\n\n")
    )
  }, character(1), USE.NAMES = FALSE)
  
  sprintf(
    "We are continuing the discussion for cluster %s (marker genes: %s%s).
    
    Previous discussion:
    %s
    
    This is round %d of the discussion. 
    
    Using the Toulmin argumentation model, please structure your response as follows:
    
    1. CLAIM: State your clear cell type prediction (e.g., 'This is a T cell')
    2. GROUNDS/DATA: Present specific observable evidence that supports your claim (e.g., marker genes expressed in this cluster)
    3. WARRANT: Explain the logical reasoning that connects your grounds to your claim (e.g., why these genes imply this cell type)
    4. BACKING: Provide additional support for your warrant (e.g., citations, references, established knowledge in the field)
    5. QUALIFIER: Indicate any conditions or limitations affecting the certainty of your claim (e.g., 'probably', 'likely', 'almost certainly')
    6. REBUTTAL: Acknowledge possible counter-arguments or exceptions (including addressing other models' predictions)
    
    Based on previous discussion, also indicate:
    - Whether you agree or disagree with any emerging consensus
    - If you've revised your previous position, explain why
    
    Format your response as:
    CELL TYPE: [your current prediction]
    GROUNDS: [specific marker genes and expression evidence supporting your claim]
    WARRANT: [logical connection between your evidence and claim]
    BACKING: [additional support for your reasoning]
    QUALIFIER: [degree of certainty - definite, probable, possible, etc.]
    REBUTTAL: [addressing counter-arguments or alternative interpretations]
    CONSENSUS STATUS: [Agree/Disagree with emerging consensus]",
    cluster_id,
    cluster_genes,
    if (!is.null(tissue_name)) sprintf(" from %s", tissue_name) else "",
    paste(discussion_history, collapse = "\n\n"),
    round_number
  )
}

#' Create prompt for the initial round of discussion
#
#
#
#
#
#' @keywords internal
create_initial_discussion_prompt <- function(cluster_id,
                                             cluster_genes,
                                             tissue_name,
                                             initial_predictions) {
  sprintf(
    "We are analyzing cluster %s with the following marker genes: %s%s
    Different models have made different predictions:
    %s
    
    Please provide your cell type prediction using the Toulmin argumentation model to structure your response:
    
    1. CLAIM: State your clear cell type prediction (e.g., 'This is a T cell')
    2. GROUNDS/DATA: Present specific observable evidence that supports your claim (e.g., marker genes expressed in this cluster)
    3. WARRANT: Explain the logical reasoning that connects your grounds to your claim (e.g., why these genes imply this cell type)
    4. BACKING: Provide additional support for your warrant (e.g., citations, references, established knowledge in the field)
    5. QUALIFIER: Indicate any conditions or limitations affecting the certainty of your claim (e.g., 'probably', 'likely', 'almost certainly')
    6. REBUTTAL: Acknowledge possible counter-arguments or exceptions (including addressing other models' predictions)
    
    Format your response as:
    CELL TYPE: [your predicted cell type]
    GROUNDS: [specific marker genes and expression evidence supporting your claim]
    WARRANT: [logical connection between your evidence and claim]
    BACKING: [additional support for your reasoning]
    QUALIFIER: [degree of certainty - definite, probable, possible, etc.]
    REBUTTAL: [addressing counter-arguments or alternative interpretations]",
    cluster_id,
    cluster_genes,
    if (!is.null(tissue_name)) sprintf(" from %s", tissue_name) else "",
    paste(
      format_named_model_responses(
        initial_predictions,
        separator = ": ",
        missing = "No prediction",
        validator = is_real_cell_type_annotation
      ),
      collapse = "\n"
    )
  )
}

#' Create prompt for standardizing cell type names
#
#
#' @keywords internal
create_standardization_prompt <- function(all_cell_types) {
  paste0(
    "I need to standardize the following cell type names to ensure consistent nomenclature while preserving biological subtypes and specificity. \n\n",
    "IMPORTANT GUIDELINES:\n",
    "1. Preserve cell subtype information: Do NOT collapse specific subtypes into general categories (e.g., DO NOT convert 'Memory B cell' to just 'B cell').\n",
    "2. Standardize expression variations: Only standardize different ways of expressing the same biological entity (e.g., 'CD4+ T cell' and 'Helper T cell' can be standardized to 'CD4+ T cell').\n",
    "3. Maintain granularity: If a cell type has specific markers or functional designations, preserve that information.\n",
    "4. Use widely accepted nomenclature: When standardizing, use the most scientifically accepted term.\n",
    "5. Be consistent with surface markers: Use consistent formatting for surface markers (e.g., CD4+, CD8+).\n\n",
    "Examples of CORRECT standardization:\n",
    "- 'Helper T cell' -> 'CD4+ T cell' (same biological entity, standard nomenclature)\n",
    "- 'CD14+ monocyte' -> 'Classical monocyte' (if they are biologically equivalent)\n",
    "- 'B-lymphocyte' -> 'B cell' (expression variation)\n\n",
    "Examples of INCORRECT standardization:\n",
    "- 'Memory B cell' -> 'B cell' (loses subtype information)\n",
    "- 'Regulatory T cell' -> 'CD4+ T cell' (loses functional subtype)\n",
    "- 'Classical monocyte' -> 'Monocyte' (loses subtype information)\n\n",
    "Please provide a standardized name for each cell type in the exact format: 'ORIGINAL: STANDARDIZED'.\n",
    "Do not add any additional text, explanations, or formatting.\n\n",
    "Here are the cell types to standardize:\n\n",
    paste(all_cell_types, collapse = "\n")
  )
}
