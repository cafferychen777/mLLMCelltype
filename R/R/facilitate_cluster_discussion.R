#' Facilitate discussion for a controversial cluster
#' @note This function uses create_initial_discussion_prompt and create_discussion_prompt from prompt_templates.R
#' @keywords internal
facilitate_cluster_discussion <- function(cluster_id,
                                          input,
                                          tissue_name,
                                          models,
                                          api_keys,
                                          initial_predictions,
                                          top_gene_count,
                                          max_rounds = 3,
                                          controversy_threshold = 0.7,
                                          entropy_threshold = 1.0,
                                          consensus_check_model = NULL,
                                          logger) {

  # Ensure cluster_id is always a string
  char_cluster_id <- as.character(cluster_id)

  # Get marker genes for this cluster
  tryCatch({
    if (inherits(input, 'list')) {
      # Check the structure of input[[char_cluster_id]]
      if (is.list(input[[char_cluster_id]]) && "genes" %in% names(input[[char_cluster_id]])) {
        # If it's a list containing a 'genes' element, extract genes and convert to string
        cluster_genes <- paste(head(input[[char_cluster_id]]$genes, top_gene_count), collapse = ",")
      } else if (is.character(input[[char_cluster_id]])) {
        # If it's already a character vector, use directly
        cluster_genes <- paste(head(input[[char_cluster_id]], top_gene_count), collapse = ",")
      } else {
        # If it's another type, try to convert to string
        cluster_genes <- paste("Cluster", char_cluster_id, "- Unable to extract specific genes")
        warning("Unable to extract genes from input for cluster ", char_cluster_id)
      }
    } else {
      # For dataframe input
      cluster_data <- input[input$cluster == char_cluster_id & input$avg_log2FC > 0, ]
      if (nrow(cluster_data) > 0 && "gene" %in% names(cluster_data)) {
        cluster_genes <- paste(head(cluster_data$gene, top_gene_count), collapse = ",")
      } else {
        cluster_genes <- paste("Cluster", char_cluster_id, "- No significant genes found")
        warning("No significant genes found for cluster ", char_cluster_id)
      }
    }
  }, error = function(e) {
    # Catch all errors and provide a default value
    cluster_genes <- paste("Cluster", char_cluster_id, "- Error extracting genes:", e$message)
    warning("Error extracting genes for cluster ", char_cluster_id, ": ", e$message)
  })

  # Initialize discussion log
  # Restructure initial_predictions if needed to extract predictions for this cluster
  structured_predictions <- list()

  for (model_name in names(initial_predictions)) {
    model_preds <- initial_predictions[[model_name]]

    # Check if model_preds is already structured by cluster_id
    if (is.list(model_preds) && !is.null(names(model_preds))) {
      # Already structured, just extract the prediction for this cluster
      structured_predictions[[model_name]] <- if (!is.null(model_preds[[char_cluster_id]])) {
        model_preds[[char_cluster_id]]
      } else {
        "Prediction_Missing"
      }
    } else if (is.character(model_preds)) {
      # Parse text lines to extract prediction for this cluster
      prediction <- "Prediction_Missing"

      # Check if there are predictions with cluster ID
      has_cluster_id_format <- FALSE

      # Process each line which should be in format: "cluster_id: cell_type"
      for (line in model_preds) {
        # Skip empty lines
        if (trimws(line) == "") next

        # Try to parse the line as "cluster_id: cell_type"
        parts <- strsplit(line, ":", fixed = TRUE)[[1]]
        if (length(parts) >= 2) {
          line_cluster_id <- trimws(parts[1])
          if (line_cluster_id == char_cluster_id) {
            cell_type <- trimws(paste(parts[-1], collapse = ":"))
            prediction <- cell_type
            has_cluster_id_format <- TRUE
            break
          }
        }
      }

      # If no prediction with cluster ID is found, try using index position
      if (!has_cluster_id_format && length(model_preds) > as.numeric(char_cluster_id)) {
        # Assume predictions are arranged in order of cluster ID
        index <- as.numeric(char_cluster_id) + 1  # Convert from 0-based to 1-based
        if (index <= length(model_preds)) {
          potential_cell_type <- trimws(model_preds[index])
          # Check if it contains ":", if so, extract the part after it
          if (grepl(":", potential_cell_type, fixed = TRUE)) {
            parts <- strsplit(potential_cell_type, ":", fixed = TRUE)[[1]]
            if (length(parts) >= 2) {
              prediction <- trimws(paste(parts[-1], collapse = ":"))
            }
          } else {
            # Does not contain ":", use directly
            prediction <- potential_cell_type
          }
        }
      }

      structured_predictions[[model_name]] <- prediction
    }
  }

  # Create the discussion log with extracted predictions
  discussion_log <- list(
    cluster_id = char_cluster_id,
    initial_predictions = structured_predictions,
    rounds = list()
  )

  # Initialize clustering discussion log file
  logger$start_cluster_discussion(char_cluster_id, tissue_name, cluster_genes)

  # First round: Initial reasoning
  first_round_prompt <- create_initial_discussion_prompt(
    cluster_id = char_cluster_id,
    cluster_genes = cluster_genes,
    tissue_name = tissue_name,
    initial_predictions = initial_predictions
  )

  # First round responses
  round1_responses <- list()
  for (model in models) {
    api_key <- get_api_key(model, api_keys)
    if (is.null(api_key)) {
      warning(sprintf("No API key found for model '%s' (provider: %s). This model will be skipped.",
                   model, get_provider(model)))
      next
    }

    response <- get_model_response(
      prompt = first_round_prompt,
      model = model,
      api_key = api_key
    )

    round1_responses[[model]] <- response
    logger$log_prediction(model, 1, response)
  }

  discussion_log$rounds[[1]] <- list(
    round_number = 1,
    responses = round1_responses
  )

  # Check consensus after first round
  # Parameters are passed to check_consensus and used in prompt template to instruct LLM # nolint
  consensus_result <- check_consensus(round1_responses, api_keys, controversy_threshold, entropy_threshold, consensus_check_model)
  logger$log_consensus_check(1, consensus_result$reached, consensus_result$consensus_proportion, consensus_result$entropy)

  # Store consensus result in discussion log
  discussion_log$rounds[[1]]$consensus_result <- consensus_result

  if (consensus_result$reached && consensus_result$consensus_proportion >= controversy_threshold && consensus_result$entropy <= entropy_threshold) {
    consensus_reached <- TRUE
    message(sprintf("Consensus reached in round 1 with consensus proportion %.2f and entropy %.2f. Stopping discussion.",
                   consensus_result$consensus_proportion, consensus_result$entropy))
    return(discussion_log)
  }

  # Additional rounds of discussion if needed
  round <- 1
  consensus_reached <- FALSE

  while (round < max_rounds && !consensus_reached) {
    round <- round + 1
    message(sprintf("\nStarting round %d of discussion...", round))

    # Create prompt that includes all previous responses
    discussion_prompt <- create_discussion_prompt(
      cluster_id = char_cluster_id,
      cluster_genes = cluster_genes,
      tissue_name = tissue_name,
      previous_rounds = discussion_log$rounds,
      round_number = round
    )

    round_responses <- list()
    for (model in models) {
      api_key <- get_api_key(model, api_keys)
      if (is.null(api_key)) {
        warning(sprintf("No API key found for model '%s' (provider: %s). This model will be skipped.",
                     model, get_provider(model)))
        next
      }

      response <- get_model_response(
        prompt = discussion_prompt,
        model = model,
        api_key = api_key
      )

      round_responses[[model]] <- response
      logger$log_prediction(model, round, response)
    }

    discussion_log$rounds[[round]] <- list(
      round_number = round,
      responses = round_responses
    )

    # Check if consensus is reached
    # Parameters are passed to check_consensus and used in prompt template to instruct LLM # nolint
    consensus_result <- check_consensus(round_responses, api_keys, controversy_threshold, entropy_threshold, consensus_check_model)
    logger$log_consensus_check(round, consensus_result$reached,
                              consensus_result$consensus_proportion, consensus_result$entropy)

    # Store consensus result in discussion log
    discussion_log$rounds[[round]]$consensus_result <- consensus_result

    # Add extracted cell types to the discussion log
    discussion_log$rounds[[round]]$extracted_cell_types <- consensus_result$extracted_cell_types

    # If we have high confidence consensus, stop the discussion
    if (consensus_result$reached && consensus_result$consensus_proportion >= controversy_threshold && consensus_result$entropy <= entropy_threshold) {
      consensus_reached <- TRUE
      message(sprintf("Consensus reached in round %d with consensus proportion %.2f and entropy %.2f. Stopping discussion.",
                     round, consensus_result$consensus_proportion, consensus_result$entropy))
    } else {
      message(sprintf("No strong consensus in round %d (consensus proportion: %.2f, entropy: %.2f). %s",
                     round, consensus_result$consensus_proportion, consensus_result$entropy,
                     if (round < max_rounds) "Continuing discussion..." else "Reached maximum rounds."))
    }

    # Only break the discussion loop if all consensus conditions are met
    if (consensus_result$reached && consensus_result$consensus_proportion >= controversy_threshold && consensus_result$entropy <= entropy_threshold) {
      break
    }
  }

  # End cluster discussion log recording
  logger$end_cluster_discussion()

  discussion_log
}
