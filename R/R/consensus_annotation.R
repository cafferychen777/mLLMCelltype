# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Generate accepted aliases for one cluster ID.
prediction_cluster_aliases <- function(cluster_id) {
  normalized <- trimws(as.character(cluster_id))
  stripped <- trimws(sub("^cluster[ _]+", "", normalized, ignore.case = TRUE))
  unique(c(
    normalized,
    stripped,
    if (nzchar(stripped)) paste("Cluster", stripped) else character(0),
    if (nzchar(stripped)) paste0("Cluster_", stripped) else character(0)
  ))
}

resolve_prediction_cluster_id <- function(raw_cluster_id, all_clusters = NULL) {
  normalized <- trimws(as.character(raw_cluster_id))
  if (!nzchar(normalized)) {
    return(NULL)
  }

  if (is.null(all_clusters)) {
    stripped <- trimws(sub("^cluster[ _]+", "", normalized, ignore.case = TRUE))
    return(if (nzchar(stripped)) stripped else normalized)
  }

  target_clusters <- unique(as.character(all_clusters))
  exact_index <- match(normalized, target_clusters)
  if (!is.na(exact_index)) {
    return(target_clusters[[exact_index]])
  }

  normalized_lower <- tolower(normalized)
  matches <- target_clusters[vapply(target_clusters, function(cluster_id) {
    normalized_lower %in% tolower(prediction_cluster_aliases(cluster_id))
  }, logical(1))]
  if (length(matches) == 1) {
    return(matches[[1]])
  }
  NULL
}

parse_prediction_line <- function(line) {
  colon_match <- regexec(
    "^\\s*(?:[-*]\\s*)?(.+?)\\s*[:\uFF1A]\\s*(.*?)\\s*$",
    line,
    perl = TRUE
  )
  colon_parts <- regmatches(line, colon_match)[[1]]
  if (length(colon_parts) == 3) {
    return(list(cluster_id = colon_parts[[2]], annotation = colon_parts[[3]]))
  }

  numbered_match <- regexec(
    "^\\s*(\\d+)\\s*[.)-]\\s+(.+?)\\s*$",
    line,
    perl = TRUE
  )
  numbered_parts <- regmatches(line, numbered_match)[[1]]
  if (length(numbered_parts) == 3) {
    return(list(cluster_id = numbered_parts[[2]], annotation = numbered_parts[[3]]))
  }
  NULL
}

#' Whether a label key denotes a cluster id (bare number or "Cluster N")
#'
#' @param key Parsed label key.
#' @return \code{TRUE} when the key is an explicit cluster reference.
#' @keywords internal
looks_like_cluster_ref <- function(key) {
  stripped <- trimws(sub("(?i)^cluster[ _]*", "", trimws(as.character(key)), perl = TRUE))
  grepl("^[0-9]+$", stripped)
}

#' Extract one line's annotation content for positional mapping
#'
#' Returns \code{NULL} only for lines that do not occupy a cluster slot: an
#' empty-value preamble/header (\code{"Notes:"}) and an explicit label for an
#' unrequested/nonexistent cluster (\code{"Cluster 999: Noise"}). Every other
#' line -- including a sentinel such as \code{"Unknown"} -- keeps its slot so
#' the line->cluster correspondence is preserved (the caller leaves a
#' non-real slot unassigned). A numbered-list ordinal (\code{"1. T cells"}) or
#' a resolvable label prefix (\code{"0: T cells"}) is stripped to its content;
#' a colon that is part of the annotation (\code{"Neurons: excitatory"}) is
#' kept whole.
#'
#' @param line Single response line.
#' @param all_clusters Character vector of requested cluster IDs.
#' @return Annotation content string, or \code{NULL} to drop the line.
#' @keywords internal
extract_positional_annotation <- function(line, all_clusters) {
  # Numbered-list ordinal ("1. T cells", "2) B cells"): the leading number is a
  # list index, not a cluster id -> keep the content after it. The delimiter
  # class excludes "-" so a hyphenated cell type ("5 - HT neurons") is not
  # mistaken for a numbered item.
  numbered <- regmatches(
    line,
    regexec("^\\s*(?:[-*]\\s*)?\\d+\\s*[.)]\\s+(.+?)\\s*$", line, perl = TRUE)
  )[[1]]
  if (length(numbered) == 2) {
    return(numbered[[2]])
  }

  parsed <- parse_prediction_line(line)
  content <- line
  if (!is.null(parsed)) {
    if (!nzchar(trimws(parsed$annotation))) {
      return(NULL)  # preamble/header line such as "Here are the annotations:"
    }
    if (!is.null(resolve_prediction_cluster_id(parsed$cluster_id, all_clusters))) {
      content <- trimws(parsed$annotation)
    } else if (looks_like_cluster_ref(parsed$cluster_id) && grepl("[:：]", line)) {
      return(NULL)  # explicit colon label for an unrequested/nonexistent cluster
    }
    # else: part of the annotation ("Neurons: excitatory", "5 - HT neurons")
  }
  content
}

#' Parse text-format model predictions into a named list
#'
#' Reads a response as explicit cluster labels (keyed by resolved cluster ID,
#' so out-of-order labels land correctly) when at least one label resolves,
#' unless an unresolved cluster-reference key coexists with incomplete coverage
#' -- that signals list ordinals misaligned against the requested clusters and
#' falls through to positional mapping. Positional mapping preserves the
#' line->cluster slot correspondence so a mid-list "Unknown" does not shift
#' later clusters, and a stray "Summary:"/"Note:" line does not hijack the parse.
#'
#' @param model_preds Character vector of prediction lines from a model
#' @param all_clusters Optional character vector of cluster IDs for positional fallback
#' @return Named list mapping cluster IDs to cell type annotations
#' @keywords internal
parse_text_predictions <- function(model_preds, all_clusters = NULL) {
  if (!is.character(model_preds)) {
    stop("model_preds must be a character vector")
  }

  lines <- unlist(strsplit(model_preds, "\n", fixed = TRUE), use.names = FALSE)
  lines <- trimws(lines[!is.na(lines) & nzchar(trimws(lines))])

  # Phase 1: read the response as explicit cluster labels, keyed by resolved
  # cluster ID. The labeled path is used when at least one label resolves,
  # UNLESS an unresolved cluster-reference key coexists with incomplete coverage
  # -- that signals list ordinals misaligned against the requested clusters
  # (e.g. a 1-based numbered list on 0-based clusters), which must fall through
  # to positional mapping. A stray word-keyed colon line (Summary:, Note:) is
  # ignored; a hallucinated out-of-range label does not spoil an otherwise
  # fully-covered labeled response.
  labeled <- list()
  resolved_clusters <- character(0)
  has_unresolved_cluster_ref <- FALSE
  for (line in lines) {
    parsed <- parse_prediction_line(line)
    if (is.null(parsed) || !nzchar(trimws(parsed$annotation))) {
      next
    }
    cluster_id <- resolve_prediction_cluster_id(parsed$cluster_id, all_clusters)
    if (!is.null(cluster_id)) {
      resolved_clusters <- union(resolved_clusters, cluster_id)
      if (is.null(labeled[[cluster_id]]) &&
          is_real_cell_type_annotation(parsed$annotation)) {
        labeled[[cluster_id]] <- trimws(parsed$annotation)
      }
    } else if (looks_like_cluster_ref(parsed$cluster_id)) {
      has_unresolved_cluster_ref <- TRUE
    }
  }

  n_clusters <- length(unique(as.character(all_clusters)))
  is_labeled <- length(resolved_clusters) > 0 &&
    !(has_unresolved_cluster_ref && length(resolved_clusters) < n_clusters)
  if (is_labeled) {
    # Return in requested-cluster order (deterministic; matches the Python twin)
    # so out-of-order labels do not surface in response order.
    if (!is.null(all_clusters)) {
      labeled <- labeled[intersect(unique(as.character(all_clusters)), names(labeled))]
    }
    return(labeled)
  }

  # Phase 2: positional fallback. Extract each line's annotation content,
  # dropping only non-slot lines (preamble, unrequested-cluster labels). Every
  # remaining line keeps its slot; a slot whose content is not a real cell type
  # (e.g. "Unknown") is left unassigned so the alignment is preserved.
  if (is.null(all_clusters)) {
    return(labeled)
  }
  target_clusters <- unique(as.character(all_clusters))
  contents <- character(0)
  for (line in lines) {
    content <- extract_positional_annotation(line, all_clusters)
    if (!is.null(content)) {
      contents <- c(contents, content)
    }
  }

  model_structured <- list()
  for (index in seq_len(min(length(target_clusters), length(contents)))) {
    if (is_real_cell_type_annotation(contents[[index]])) {
      model_structured[[target_clusters[[index]]]] <- contents[[index]]
    }
  }
  model_structured
}

structure_model_predictions <- function(model_preds, all_clusters) {
  target_clusters <- unique(as.character(all_clusters))
  prediction_names <- names(model_preds)
  has_explicit_names <- !is.null(prediction_names) &&
    any(!is.na(prediction_names) & nzchar(trimws(prediction_names)))

  if (has_explicit_names && (is.list(model_preds) || is.character(model_preds))) {
    structured <- list()
    for (index in seq_along(model_preds)) {
      raw_name <- prediction_names[[index]]
      if (is.na(raw_name) || !nzchar(trimws(raw_name))) {
        next
      }

      cluster_id <- resolve_prediction_cluster_id(raw_name, target_clusters)
      annotation <- model_preds[[index]]
      is_valid_annotation <- is_real_cell_type_annotation(annotation)
      if (!is.null(cluster_id) &&
          is_valid_annotation &&
          is.null(structured[[cluster_id]])) {
        structured[[cluster_id]] <- trimws(annotation)
      }
    }
    return(structured)
  }

  if (is.character(model_preds)) {
    return(parse_text_predictions(model_preds, target_clusters))
  }

  list()
}

structure_cluster_predictions <- function(initial_predictions, all_clusters,
                                          cluster_id) {
  cluster_id <- .normalize_required_string(as.character(cluster_id), "cluster_id")
  structured_predictions <- list()

  for (model_name in names(initial_predictions)) {
    parsed <- structure_model_predictions(
      initial_predictions[[model_name]],
      all_clusters
    )
    structured_predictions[[model_name]] <- if (!is.null(parsed[[cluster_id]])) {
      parsed[[cluster_id]]
    } else {
      "Prediction_Missing"
    }
  }

  structured_predictions
}

build_discussion_cache_context <- function(input, cluster_id, models, api_keys,
                                           initial_predictions,
                                           max_discussion_rounds,
                                           controversy_threshold,
                                           entropy_threshold,
                                           consensus_check_model,
                                           base_urls) {
  all_cluster_ids <- canonical_cluster_ids(input)
  current_predictions <- structure_cluster_predictions(
    initial_predictions,
    all_cluster_ids,
    cluster_id
  )
  request_models <- unique(c(
    models,
    prepare_models_list(consensus_check_model)
  ))
  request_context <- lapply(request_models, function(model) {
    credential <- .resolve_model_api_key(model, api_keys)
    base_url <- if (credential$source == "none") {
      NULL
    } else {
      resolve_provider_base_url(credential$provider, base_urls)
    }
    model_config <- NULL
    model_name <- tolower(trimws(model))
    if (exists(model_name, envir = custom_models, inherits = FALSE)) {
      model_config <- get(model_name, envir = custom_models)$config
    }

    list(
      model = model,
      provider = credential$provider,
      credential_source = credential$source,
      base_url = base_url,
      model_config = model_config
    )
  })

  list(
    initial_predictions = current_predictions,
    max_discussion_rounds = max_discussion_rounds,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    consensus_check_model = consensus_check_model,
    requests = request_context
  )
}

align_model_predictions <- function(model_preds, all_clusters) {
  target_clusters <- unique(as.character(all_clusters))
  structured <- structure_model_predictions(model_preds, target_clusters)
  aligned <- vapply(target_clusters, function(cluster_id) {
    annotation <- structured[[cluster_id]]
    if (!is_real_cell_type_annotation(annotation)) NA_character_ else annotation
  }, character(1))
  names(aligned) <- target_clusters
  aligned
}

#' Get initial predictions from all models
#'
#' This function retrieves initial cell type predictions from all specified models.
#' It is an internal helper function used by the interactive_consensus_annotation function.
#'
#' @keywords internal
get_initial_predictions <- function(input, tissue_name, models, api_keys, top_gene_count, base_urls = NULL) {
  log_info("Phase 1: Getting initial predictions from all models...", list(
    models_count = length(models),
    models = models
  ))
  message("\nPhase 1: Getting initial predictions from all models...")

  # Initialize tracking variables
  individual_predictions <- list()
  successful_models <- character(0)

  # Get predictions from each model
  for (model in models) {
    provider <- tryCatch({
      get_provider(model)
    }, error = function(e) {
      log_warn("Failed to resolve provider for model", list(model = model, error = e$message))
      "unknown"
    })

    api_key <- tryCatch({
      get_api_key(model, api_keys)
    }, error = function(e) {
      log_warn("Failed to resolve API key", list(model = model, error = e$message))
      NULL
    })

    if (is.null(api_key)) {
      warning_msg <- sprintf("No API key found for model '%s' (provider: %s). This model will be skipped.",
                            model, provider)
      warning(warning_msg)
      log_warn(warning_msg, list(model = model, provider = provider))
      next
    }

    tryCatch({
      predictions <- annotate_cell_types(
        input = input,
        tissue_name = tissue_name,
        model = model,
        api_key = api_key,
        top_gene_count = top_gene_count,
        base_urls = base_urls
      )
      individual_predictions[[model]] <- predictions
      successful_models <- c(successful_models, model)
    }, error = function(e) {
      warning_msg <- sprintf("Failed to get predictions from %s: %s", model, e$message)
      warning(warning_msg)
      log_warn(warning_msg, list(model = model, error = e$message))
    })
  }

  if (length(successful_models) == 0) {
    stop("No models successfully completed predictions. Please check API keys and model availability.")
  }

  return(list(
    individual_predictions = individual_predictions,
    successful_models = successful_models
  ))
}

#' Identify controversial clusters based on consensus analysis
#'
#
#
#
#
#
#' @keywords internal
identify_controversial_clusters <- function(input, individual_predictions, controversy_threshold, entropy_threshold, api_keys, consensus_check_model = NULL, base_urls = NULL) {
  clusters <- canonical_cluster_ids(input)

  log_info("Phase 2: Identifying controversial clusters...", list(
    clusters_count = length(clusters),
    entropy_threshold = entropy_threshold,
    controversy_threshold = controversy_threshold
  ))
  message("\nPhase 2: Identifying controversial clusters...")

  # Initialize consensus tracking
  consensus_results <- list()
  controversial_clusters <- character(0)
  final_annotations <- list()

  # Get all cluster IDs for positional fallback in text parsing
  all_clusters <- as.character(clusters)

  # Restructure individual_predictions to be indexed by cluster_id
  structured_predictions <- list()

  for (model_name in names(individual_predictions)) {
    model_preds <- individual_predictions[[model_name]]
    structured_predictions[[model_name]] <- structure_model_predictions(
      model_preds,
      all_clusters
    )
  }

  for (cluster_id in clusters) {
    # Use original cluster ID for log output, no conversion needed
    log_info(sprintf("Analyzing cluster %s...", cluster_id), list(cluster_id = cluster_id))
    message(sprintf("\nAnalyzing cluster %s...", cluster_id))

    # Get predictions for this cluster from each model
    cluster_predictions <- sapply(structured_predictions, function(x) {
      if (is.null(x[[as.character(cluster_id)]])) NA else x[[as.character(cluster_id)]]
    })
    valid_predictions <- cluster_predictions[!is.na(cluster_predictions)]

    if (length(valid_predictions) == 0) {
      log_warn(sprintf("No valid predictions for cluster %s. Marking as controversial.", cluster_id), list(cluster_id = cluster_id))
      message(sprintf("No valid predictions for cluster %s. Marking as controversial.", cluster_id))
      controversial_clusters <- c(controversial_clusters, as.character(cluster_id))
      next
    }

    # Calculate agreement score
    initial_consensus <- check_consensus(valid_predictions, api_keys, controversy_threshold, entropy_threshold, consensus_check_model, base_urls)
    consensus_results[[as.character(cluster_id)]] <- initial_consensus

    # If no consensus is reached or the consensus metrics indicate high uncertainty, mark it as controversial.
    # Use both consensus proportion and entropy for decision making
    if (!initial_consensus$reached ||
        initial_consensus$consensus_proportion < controversy_threshold ||
        initial_consensus$entropy > entropy_threshold) {

      log_info(sprintf("Cluster %s marked as controversial", cluster_id), list(
        cluster_id = cluster_id,
        reached_consensus = initial_consensus$reached,
        consensus_proportion = initial_consensus$consensus_proportion,
        entropy = initial_consensus$entropy
      ))

      message(sprintf("Cluster %s marked as controversial (reached: %s, consensus proportion: %.2f, entropy: %.2f)",
                     cluster_id, initial_consensus$reached,
                     initial_consensus$consensus_proportion, initial_consensus$entropy))

      controversial_clusters <- c(controversial_clusters, as.character(cluster_id))
    } else {
      # Process non-controversial clusters
      final_annotations[[as.character(cluster_id)]] <- select_best_prediction(initial_consensus, valid_predictions)

      log_info(sprintf("Consensus reached for cluster %s", cluster_id), list(
        cluster_id = cluster_id,
        consensus_proportion = initial_consensus$consensus_proportion,
        entropy = initial_consensus$entropy,
        selected_cell_type = final_annotations[[as.character(cluster_id)]]
      ))

      message(sprintf("Consensus reached for cluster %s (consensus proportion: %.2f, entropy: %.2f, selected: %s)",
                     cluster_id, initial_consensus$consensus_proportion,
                     initial_consensus$entropy, final_annotations[[as.character(cluster_id)]]))
    }
  }

  return(list(
    consensus_results = consensus_results,
    controversial_clusters = controversial_clusters,
    final_annotations = final_annotations
  ))
}

#' Select the best prediction from consensus results
#'
#
#
#
#' @keywords internal
select_best_prediction <- function(consensus_result, valid_predictions) {
  majority <- clean_annotation(consensus_result$majority_prediction)

  # Accept the consensus check result if it is a real cell type
  if (!identical(majority, "Unknown")) {
    return(majority)
  }

  # Fallback: pick the most frequent real prediction from the models
  cleaned_predictions <- vapply(
    as.list(valid_predictions),
    clean_annotation,
    character(1)
  )
  real_predictions <- cleaned_predictions[cleaned_predictions != "Unknown"]
  if (length(real_predictions) == 0) {
    return("Unknown")
  }

  prediction_counts <- table(real_predictions)
  max_count <- max(prediction_counts)
  most_common <- names(prediction_counts[prediction_counts == max_count])

  if (length(most_common) == 1) {
    return(most_common)
  }
  # Preserve caller model priority when the vote is tied.
  real_predictions[real_predictions %in% most_common][[1]]
}

#' Process controversial clusters through discussion
#'
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#' @keywords internal
process_controversial_clusters <- function(controversial_clusters, input, tissue_name,
                                          successful_models, api_keys, individual_predictions,
                                          top_gene_count, controversy_threshold, entropy_threshold, max_discussion_rounds,
                                          cache_manager, use_cache, consensus_check_model = NULL, force_rerun = FALSE,
                                          base_urls = NULL) {

  if (length(controversial_clusters) == 0) {
    log_info("No controversial clusters found. All clusters have reached consensus.")
    message("\nNo controversial clusters found. All clusters have reached consensus.")
    return(list(
      discussion_logs = list(),
      final_annotations = list()
    ))
  }

  log_info(sprintf("Phase 3: Starting discussions for %d controversial clusters...",
                   length(controversial_clusters)), list(
    controversial_count = length(controversial_clusters),
    clusters = controversial_clusters
  ))
  message(sprintf("\nPhase 3: Starting discussions for %d controversial clusters...",
                 length(controversial_clusters)))

  discussion_logs <- list()
  final_annotations <- list()

  for (cluster_id in controversial_clusters) {
    # Ensure cluster_id is a string type
    char_cluster_id <- as.character(cluster_id)
    log_info(sprintf("Starting discussion for cluster %s...", char_cluster_id), list(
      cluster_id = char_cluster_id
    ))
    message(sprintf("\nStarting discussion for cluster %s...", char_cluster_id))

    # Generate cache key once (reused for both lookup and save)
    cache_key <- if (use_cache) {
      discussion_context <- build_discussion_cache_context(
        input = input,
        cluster_id = char_cluster_id,
        models = successful_models,
        api_keys = api_keys,
        initial_predictions = individual_predictions,
        max_discussion_rounds = max_discussion_rounds,
        controversy_threshold = controversy_threshold,
        entropy_threshold = entropy_threshold,
        consensus_check_model = consensus_check_model,
        base_urls = base_urls
      )
      cache_manager$generate_key(
        input,
        successful_models,
        char_cluster_id,
        tissue_name,
        top_gene_count,
        discussion_context = discussion_context
      )
    } else {
      NULL
    }

    # Check cache
    cached_result <- NULL
    if (use_cache && !force_rerun) {
      log_debug(sprintf("Cache check for cluster %s", char_cluster_id),
                list(cluster_id = char_cluster_id, cache_key = cache_key))

      cached_result <- cache_manager$load_from_cache(cache_key)
      if (!is.null(cached_result) && is_valid_consensus_cache_data(cached_result)) {
        log_info(sprintf("Loading cached result for cluster %s", char_cluster_id), list(
          cluster_id = char_cluster_id,
          cache_key = cache_key
        ))
        message(sprintf("Loading cached result for cluster %s", char_cluster_id))
      } else if (!is.null(cached_result)) {
        log_warn("Ignoring invalid cached discussion result", list(
          cluster_id = char_cluster_id,
          cache_key = cache_key
        ))
        cached_result <- NULL
      }
    } else if (force_rerun) {
      log_info(sprintf("Force rerun enabled, skipping cache for cluster %s", char_cluster_id))
    }

    # Use cached results or perform discussion
    if (!is.null(cached_result)) {
      discussion_result <- cached_result$discussion_log
      final_annotation <- cached_result$annotation

      log_info(sprintf("Using cached result for cluster %s", char_cluster_id), list(
        cluster_id = char_cluster_id
      ))
      message(sprintf("Using cached result for cluster %s", char_cluster_id))
    } else {
      # Perform discussion
      discussion_result <- facilitate_cluster_discussion(
        cluster_id = char_cluster_id,
        input = input,
        tissue_name = tissue_name,
        models = successful_models,
        api_keys = api_keys,
        initial_predictions = individual_predictions,
        top_gene_count = top_gene_count,
        max_rounds = max_discussion_rounds,
        controversy_threshold = controversy_threshold,
        entropy_threshold = entropy_threshold,
        consensus_check_model = consensus_check_model,
        base_urls = base_urls
      )

      # Find the last round that has a consensus_result (the last round may
      # lack one if it broke early due to insufficient valid responses)
      final_prediction <- NULL
      for (r in rev(seq_along(discussion_result$rounds))) {
        cr <- discussion_result$rounds[[r]]$consensus_result
        if (!is.null(cr)) {
          final_prediction <- cr$majority_prediction
          break
        }
      }

      # Extract and clean majority_prediction
      final_annotation <- clean_annotation(final_prediction)

      # Save to cache
      if (use_cache) {
        cache_data <- list(
          annotation = final_annotation,
          discussion_log = discussion_result,
          is_controversial = TRUE
        )
        cache_saved <- cache_manager$save_to_cache(cache_key, cache_data)
        if (isTRUE(cache_saved)) {
          log_info(sprintf("Saved result to cache for cluster %s", char_cluster_id), list(
            cluster_id = char_cluster_id
          ))
        }
      }
    }

    # Ensure cluster_id in discussion_result is a string type
    if (!is.null(discussion_result) && !is.character(discussion_result$cluster_id)) {
      discussion_result$cluster_id <- char_cluster_id
    }

    discussion_logs[[char_cluster_id]] <- discussion_result
    final_annotations[[char_cluster_id]] <- final_annotation

    log_info(sprintf("Completed discussion for cluster %s", char_cluster_id), list(
      cluster_id = char_cluster_id
    ))
    message(sprintf("Completed discussion for cluster %s", char_cluster_id))
  }

  return(list(
    discussion_logs = discussion_logs,
    final_annotations = final_annotations
  ))
}

#' Clean annotation text by removing prefixes and extra whitespace
#'
#
#
#' @keywords internal
clean_annotation <- function(annotation) {
  is_text_scalar <- is.character(annotation) &&
    length(annotation) == 1 &&
    !is.na(annotation)
  if (!is_text_scalar) {
    return("Unknown")
  }

  # Remove numbered prefixes like "1. ", "1: ", "1- ", etc.
  annotation <- gsub("^\\d+[\\.:\\-\\s]+\\s*", "", annotation)
  # Remove "CELL TYPE:" prefix
  annotation <- gsub(
    "^CELL\\s*TYPE[\\s:]*",
    "",
    annotation,
    ignore.case = TRUE
  )
  # Final trim of whitespace
  annotation <- trimws(annotation)

  # Normalize sentinel values to a user-friendly fallback
  if (!is_real_cell_type_annotation(annotation)) {
    return("Unknown")
  }

  return(annotation)
}

#' Combine results from all phases of consensus annotation
#'
#
#
#
#
#' @keywords internal
combine_results <- function(initial_results, controversy_results, discussion_results) {
  # Start with non-controversial cluster annotations
  final_annotations <- lapply(
    controversy_results$final_annotations,
    clean_annotation
  )

  # Merge controversial cluster annotations (already cleaned by process_controversial_clusters)
  for (cluster_id in names(discussion_results$final_annotations)) {
    char_cluster_id <- as.character(cluster_id)
    annotation <- discussion_results$final_annotations[[char_cluster_id]]
    final_annotations[[char_cluster_id]] <- clean_annotation(annotation)
  }

  result <- list(
    initial_results = list(
      individual_predictions = initial_results$individual_predictions,
      consensus_results = controversy_results$consensus_results,
      controversial_clusters = controversy_results$controversial_clusters
    ),
    final_annotations = final_annotations,
    controversial_clusters = controversy_results$controversial_clusters,
    discussion_logs = discussion_results$discussion_logs,
    session_id = get_logger()$session_id
  )

  # Backward-compatible aliases.
  result$voting_results <- result$initial_results
  result$discussion_results <- result$discussion_logs
  result$final_consensus <- result$final_annotations

  result
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

# Constants are now defined as function parameters

#' Interactive consensus building for cell type annotation
#'
#' This function implements an interactive voting and discussion mechanism where multiple LLMs
#' collaborate to reach a consensus on cell type annotations, particularly focusing on
#' clusters with low agreement. The process includes:
#' 1. Initial voting by all LLMs
#' 2. Identification of controversial clusters
#' 3. Detailed discussion for controversial clusters
#' 4. Final summary by a designated LLM (default: Claude)
#'
#' @param input Either a data frame from Seurat's FindAllMarkers() function containing 
#'   differential gene expression results (must have columns: 'cluster', 'gene', 
#'   and 'avg_log2FC'), or a list where each element is either a character vector
#'   of genes or a list containing a `genes` field.
#' @param tissue_name Character string specifying the tissue type for context-aware
#'   cell type annotation (e.g., 'human PBMC', 'mouse brain'). Required.
#' @param models Character vector of model names to use for consensus annotation. 
#'   Minimum 2 models required. Supports models from OpenAI, Anthropic, DeepSeek, 
#'   Google, Alibaba, Stepfun, Zhipu, MiniMax, X.AI, and OpenRouter.
#' @param api_keys Named, non-empty list of API keys. Can use provider names as keys
#'   (e.g., "openai", "anthropic") or model names as keys (e.g., "gpt-5").
#' @param top_gene_count Integer specifying the number of top marker genes to use 
#'   for annotation per cluster (default: 10).
#' @param controversy_threshold Numeric value between 0 and 1 for consensus proportion 
#'   threshold. Clusters below this threshold are considered controversial (default: 0.7).
#' @param entropy_threshold Numeric value for entropy threshold. Higher entropy 
#'   indicates more disagreement among models (default: 1.0).
#' @param max_discussion_rounds Integer specifying maximum number of discussion rounds 
#'   for controversial clusters (default: 3).
#' @param consensus_check_model Character string specifying which model to use for
#'   consensus checking. If NULL, uses the first model that succeeds during initial annotation.
#' @param log_dir Character scalar specifying directory for log files (default: "logs").
#'   This function reinitializes the session logger with this directory at the start
#'   of each call.
#' @param cache_dir Character string or NULL. Cache directory for storing results. 
#'   NULL uses system cache, "local" uses current directory, "temp" uses temporary 
#'   directory, or specify custom path.
#' @param use_cache Logical indicating whether to use caching (default: TRUE).
#' @param base_urls Named list or character string specifying custom API base URLs. 
#'   Useful for proxies or alternative endpoints. If NULL, uses official endpoints.
#' @param clusters_to_analyze Character or numeric vector specifying which clusters 
#'   to analyze. If NULL (default), all clusters are analyzed.
#' @param force_rerun Logical indicating whether to force rerun of all specified 
#'   clusters, ignoring cache. Only affects controversial cluster discussions 
#'   (default: FALSE).
#'
#' @return A list containing:
#'   \itemize{
#'     \item \code{initial_results}: Initial voting results, consensus checks, and controversial cluster IDs
#'     \item \code{final_annotations}: Final annotations keyed by cluster ID
#'     \item \code{controversial_clusters}: Clusters identified as controversial
#'     \item \code{discussion_logs}: Detailed discussion logs for controversial clusters
#'     \item \code{session_id}: Logger session identifier
#'     \item \code{voting_results}: Backward-compatible alias of \code{initial_results}
#'     \item \code{discussion_results}: Backward-compatible alias of \code{discussion_logs}
#'     \item \code{final_consensus}: Backward-compatible alias of \code{final_annotations}
#'   }
#' @importFrom stats setNames
#' @export
interactive_consensus_annotation <- function(input,
                                           tissue_name,
                                           models = c("claude-opus-4-7",
                                                     "gpt-5.5",
                                                     "gemini-3.1-pro-preview",
                                                     "deepseek-v4-flash",
                                                     "grok-4.3"),
                                           api_keys,
                                           top_gene_count = 10,
                                           controversy_threshold = 0.7,
                                           entropy_threshold = 1.0,
                                           max_discussion_rounds = 3,
                                           consensus_check_model = NULL,
                                           log_dir = "logs",
                                           cache_dir = NULL,
                                           use_cache = TRUE,
                                           base_urls = NULL,
                                           clusters_to_analyze = NULL,
                                           force_rerun = FALSE) {
  tissue_name <- .normalize_required_string(tissue_name, "tissue_name")
  models <- .normalize_model_vector(models, minimum_count = 2L)
  api_keys <- .normalize_api_keys(api_keys)
  top_gene_count <- .normalize_top_gene_count(top_gene_count)
  controversy_threshold <- .normalize_probability(
    controversy_threshold,
    "controversy_threshold"
  )
  entropy_threshold <- .normalize_nonnegative_number(
    entropy_threshold,
    "entropy_threshold"
  )
  max_discussion_rounds <- .normalize_positive_integer(
    max_discussion_rounds,
    "max_discussion_rounds"
  )
  log_dir <- .normalize_required_string(log_dir, "log_dir")
  use_cache <- .normalize_flag(use_cache, "use_cache")
  force_rerun <- .normalize_flag(force_rerun, "force_rerun")
  if (!is.null(consensus_check_model)) {
    consensus_check_model <- .normalize_required_string(
      consensus_check_model,
      "consensus_check_model"
    )
  }

  if (!is.null(clusters_to_analyze)) {
    selector_is_valid <- (is.character(clusters_to_analyze) ||
      is.numeric(clusters_to_analyze)) &&
      length(clusters_to_analyze) > 0 &&
      !anyNA(clusters_to_analyze)
    if (!selector_is_valid) {
      stop("clusters_to_analyze must be a non-empty character or numeric vector")
    }
    clusters_to_analyze <- unique(trimws(as.character(clusters_to_analyze)))
    if (any(!nzchar(clusters_to_analyze))) {
      stop("clusters_to_analyze must not contain empty cluster IDs")
    }
  }

  cluster_name_map <- NULL

  # Normalize list input to a canonical cluster->genes mapping to keep
  # contract consistent with annotate_cell_types/create_annotation_prompt.
  if (is.list(input) && !is.data.frame(input)) {
    original_names <- names(input)
    normalized_input <- normalize_cluster_gene_list(input)
    if (!is.null(original_names)) {
      cluster_name_map <- setNames(names(normalized_input), original_names)
    }
    input <- lapply(normalized_input, function(genes) list(genes = genes))
  }

  prompt_metadata <- create_annotation_prompt(input, tissue_name, top_gene_count)
  available_clusters <- names(prompt_metadata$gene_lists)
  if (is.list(input) && !is.data.frame(input)) {
    input <- input[available_clusters]
  }

  initialize_logger(log_dir)

  # Initialize cache manager
  cache_manager <- CacheManager$new(cache_dir)
  
  # Get actual cache directory path (important!)
  actual_cache_dir <- cache_manager$get_cache_dir()

  # Log cache settings - use actual path
  if (use_cache && !force_rerun) {
    cache_msg <- sprintf("Cache enabled. Using cache directory: %s", actual_cache_dir)
    log_info(cache_msg, list(cache_dir = actual_cache_dir))
    message(cache_msg)
  } else if (force_rerun) {
    log_info("Force rerun enabled, cache will be ignored for controversial clusters")
    message("Force rerun enabled. Cache will be ignored for controversial clusters.")
  } else {
    log_info("Cache disabled")
    message("Cache disabled.")
  }

  # Filter clusters if clusters_to_analyze is specified
  if (!is.null(clusters_to_analyze)) {
    # If list input names were normalized, accept either original or normalized IDs.
    if (!is.null(cluster_name_map)) {
      mapped_ids <- ifelse(
        clusters_to_analyze %in% names(cluster_name_map),
        cluster_name_map[clusters_to_analyze],
        clusters_to_analyze
      )
      clusters_to_analyze <- as.character(mapped_ids)
    }
    
    # Check which requested clusters exist
    valid_clusters <- clusters_to_analyze[clusters_to_analyze %in% available_clusters]
    invalid_clusters <- clusters_to_analyze[!clusters_to_analyze %in% available_clusters]
    
    # Warn about non-existent clusters
    if (length(invalid_clusters) > 0) {
      warning(sprintf("The following cluster IDs were not found in the input: %s",
                     paste(invalid_clusters, collapse = ", ")))
      log_warn("Specified cluster IDs not found in input", list(invalid_clusters = invalid_clusters))
    }
    
    # Stop if no valid clusters
    if (length(valid_clusters) == 0) {
      stop("None of the specified clusters exist in the input data.")
    }
    
    # Filter input based on type
    if (is.list(input) && !is.data.frame(input)) {
      # For list input, subset by names
      input <- input[valid_clusters]
    } else {
      # For dataframe input, filter rows
      normalized_input_clusters <- trimws(as.character(input$cluster))
      input <- input[normalized_input_clusters %in% valid_clusters, , drop = FALSE]
    }
    
    # Log the filtering
    log_info(sprintf("Filtered to analyze %d clusters: %s", 
                    length(valid_clusters), 
                    paste(valid_clusters, collapse = ", ")))
    message(sprintf("Analyzing %d specified clusters: %s", 
                   length(valid_clusters), 
                   paste(valid_clusters, collapse = ", ")))
  }

  # Phase 1: Get initial predictions from all models
  initial_results <- get_initial_predictions(
    input = input,
    tissue_name = tissue_name,
    models = models,
    api_keys = api_keys,
    top_gene_count = top_gene_count,
    base_urls = base_urls
  )

  # Phase 2: Identify controversial clusters
  # If consensus_check_model is NULL, use the first model that succeeded in
  # Phase 1 (not models[1], which may have failed)
  if (is.null(consensus_check_model)) {
    consensus_check_model <- initial_results$successful_models[1]
    log_info("No consensus_check_model specified, using first successful model",
             list(consensus_check_model = consensus_check_model))
  }

  controversy_results <- identify_controversial_clusters(
    input = input,
    individual_predictions = initial_results$individual_predictions,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    api_keys = api_keys,
    consensus_check_model = consensus_check_model,
    base_urls = base_urls
  )

  # Phase 3: Process controversial clusters through discussion
  discussion_results <- process_controversial_clusters(
    controversial_clusters = controversy_results$controversial_clusters,
    input = input,
    tissue_name = tissue_name,
    successful_models = initial_results$successful_models,
    api_keys = api_keys,
    individual_predictions = initial_results$individual_predictions,
    top_gene_count = top_gene_count,
    controversy_threshold = controversy_threshold,
    entropy_threshold = entropy_threshold,
    max_discussion_rounds = max_discussion_rounds,
    cache_manager = cache_manager,
    use_cache = use_cache,
    consensus_check_model = consensus_check_model,
    force_rerun = force_rerun,
    base_urls = base_urls
  )

  # Combine results from all phases
  final_results <- combine_results(
    initial_results = initial_results,
    controversy_results = controversy_results,
    discussion_results = discussion_results
  )

  # Print summary of consensus building process
  print_consensus_summary(final_results)

  # Return results
  return(final_results)
}
