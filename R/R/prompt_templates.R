#' Prompt templates for mLLMCelltype
#' 
#' This file contains all prompt template functions used in mLLMCelltype.
#' These functions create various prompts for different stages of the cell type annotation process.
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

  extract_item_genes <- function(item) {
    if (is.list(item) && "genes" %in% names(item)) {
      return(as.character(item$genes))
    }
    if (is.character(item)) {
      return(item)
    }
    stop("When input is a list, each element must be a character vector of genes or a list containing a 'genes' field")
  }

  names_vec <- names(input)
  gene_vectors <- lapply(input, extract_item_genes)

  if (is.null(names_vec)) {
    canonical_names <- as.character(seq_along(gene_vectors) - 1)
  } else {
    canonical_names <- names_vec
  }

  if (anyDuplicated(canonical_names)) {
    stop("Duplicate cluster IDs detected after normalization")
  }

  names(gene_vectors) <- canonical_names
  gene_vectors
}

#' Create prompt for cell type annotation
#'
#' @param input Either a data frame from Seurat's FindAllMarkers() or a list for each cluster
#'   where each element is either a character vector of genes or a list containing a `genes` field
#'   Cluster IDs in named inputs are preserved as-is; unnamed list input receives sequential IDs starting at "0".
#' @param tissue_name Tissue context for the annotation (e.g., 'human PBMC', 'mouse brain')
#' @param top_gene_count Number of top genes to use per cluster when input is from Seurat. Default: 10
#'
#' @return Character string containing the formatted prompt
#' @importFrom magrittr "%>%"
#' @export
create_annotation_prompt <- function(input, tissue_name, top_gene_count = 10) {
  if (is.list(input) && !is.data.frame(input)) {
    normalized_input <- normalize_cluster_gene_list(input)

    gene_lists <- list()

    for (cluster_id in names(normalized_input)) {
      genes <- normalized_input[[cluster_id]]
      gene_lists[[cluster_id]] <- paste(genes, collapse = ", ")
    }
    
    expected_count <- length(normalized_input)
  } else if (is.data.frame(input)) {
    # Process Seurat differential gene table
    # Cluster IDs are preserved as-is from the input data
    markers <- input %>%
      group_by(cluster) %>%
      slice_max(avg_log2FC, n = top_gene_count, with_ties = FALSE) %>%
      group_split()

    gene_lists <- list()
    for (marker_group in markers) {
      cluster_id <- marker_group$cluster[1]
      gene_lists[[as.character(cluster_id)]] <- paste(marker_group$gene, collapse = ',')
    }

    expected_count <- length(unique(input$cluster))
  } else {
    stop("Input must be either a data.frame (from Seurat) or a list of gene lists")
  }
  
  # Create formatted lines from gene_lists using actual names/indices
  cluster_names <- names(gene_lists)
  formatted_lines <- vapply(cluster_names, function(name) {
    paste0(name, ": ", gene_lists[[name]])
  }, character(1), USE.NAMES = FALSE)

  # Sort numerically if all names are numeric
  if (all(!is.na(suppressWarnings(as.numeric(cluster_names))))) {
    formatted_lines <- formatted_lines[order(as.numeric(cluster_names))]
  }

  prompt <- paste0("You are a cell type annotation expert. Below are marker genes for different cell clusters in ", 
                  tissue_name, ".\n\n",
                  paste(formatted_lines, collapse = "\n"),
                  "\n\nFor each numbered cluster, provide only the cell type name in a new line, without any explanation.")
  
  return(list(
    prompt = prompt,
    expected_count = expected_count,
    gene_lists = gene_lists
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
    "1. Consider predictions as matching if they refer to the same cell type, ignoring differences in:",
    "   - Formatting (e.g., 'NK cells' vs 'Natural Killer cells')",
    "   - Capitalization",
    "   - Additional qualifiers (e.g., 'activated', 'mature', etc.)",
    "2. Group predictions that refer to the same cell type",
    "3. If any prediction is 'Unknown' or 'Unclear', treat it as a separate group",
    "",
    "CALCULATE THE FOLLOWING METRICS:",
    "1. Consensus Proportion = Number of models supporting the majority prediction / Total number of models",
    "2. Shannon Entropy = -sum(p_i * log2(p_i)) where p_i is the proportion of models predicting each unique cell type",
    sprintf("3. Determine if consensus is reached (Consensus Proportion > %s AND Entropy <= %s)", 
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
  discussion_history <- sapply(previous_rounds, function(round) {
    responses <- round$responses
    paste(sprintf("Round %d:\n%s\n",
                  round$round_number,
                  paste(sprintf("%s:\n%s\n", 
                                names(responses),
                                responses), 
                        collapse = "\n")))
  })
  
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
    paste(sapply(names(initial_predictions), function(model_name) {
      pred <- initial_predictions[[model_name]]
      # Check if pred is a list with named elements
      if (is.list(pred) && !is.null(names(pred))) {
        # If it's a structured list, try to extract the prediction for this cluster
        cell_type <- if (!is.null(pred[[as.character(cluster_id)]])) {
          pred[[as.character(cluster_id)]]
        } else {
          "No prediction"
        }
      } else if (is.character(pred)) {
        # If it's a character vector, try to find the line for this cluster
        cell_type <- "No prediction"
        
        # Check if there are predictions with cluster ID
        has_cluster_id_format <- FALSE
        for (line in pred) {
          if (trimws(line) == "") next
          parts <- strsplit(line, ":", fixed = TRUE)[[1]]
          if (length(parts) >= 2 && trimws(parts[1]) == as.character(cluster_id)) {
            cell_type <- trimws(paste(parts[-1], collapse = ":"))
            has_cluster_id_format <- TRUE
            break
          }
        }
        
      } else {
        cell_type <- "No prediction"
      }
      sprintf("%s: %s", model_name, cell_type)
    }), collapse = "\n")
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
