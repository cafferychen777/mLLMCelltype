#' Filter out error responses from model round responses
#' @keywords internal
filter_valid_responses <- function(responses, cluster_id, round = NULL) {
  valid <- list()
  for (model_name in names(responses)) {
    response <- responses[[model_name]]
    if (!is.null(response) &&
        (!is.character(response) ||
         (is.character(response) && !any(grepl("^Error:", response))))) {
      valid[[model_name]] <- response
    } else {
      round_info <- if (!is.null(round)) sprintf(" in round %d", round) else ""
      log_warn(sprintf("Model %s failed to provide valid response for cluster %s%s",
                      model_name, cluster_id, round_info))
    }
  }
  valid
}

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
                                          base_urls = NULL) {

  # Ensure cluster_id is always a string
  char_cluster_id <- as.character(cluster_id)

  # Get marker genes for this cluster
  cluster_genes <- tryCatch({
    if (inherits(input, 'list')) {
      # Check the structure of input[[char_cluster_id]]
      if (is.list(input[[char_cluster_id]]) && "genes" %in% names(input[[char_cluster_id]])) {
        # If it's a list containing a 'genes' element, extract genes and convert to string
        paste(head(input[[char_cluster_id]]$genes, top_gene_count), collapse = ",")
      } else if (is.character(input[[char_cluster_id]])) {
        # If it's already a character vector, use directly
        paste(head(input[[char_cluster_id]], top_gene_count), collapse = ",")
      } else {
        # If it's another type, provide fallback
        warning("Unable to extract genes from input for cluster ", char_cluster_id)
        paste("Cluster", char_cluster_id, "- Unable to extract specific genes")
      }
    } else {
      # For dataframe input
      cluster_data <- input[input$cluster == char_cluster_id & input$avg_log2FC > 0, ]
      if (nrow(cluster_data) > 0 && "gene" %in% names(cluster_data)) {
        paste(head(cluster_data$gene, top_gene_count), collapse = ",")
      } else {
        warning("No significant genes found for cluster ", char_cluster_id)
        paste("Cluster", char_cluster_id, "- No significant genes found")
      }
    }
  }, error = function(e) {
    warning("Error extracting genes for cluster ", char_cluster_id, ": ", e$message)
    paste("Cluster", char_cluster_id, "- Error extracting genes:", e$message)
  })

  # Extract predictions for this cluster from each model
  # Uses the shared parse_text_predictions() helper for text-format responses
  structured_predictions <- list()

  for (model_name in names(initial_predictions)) {
    model_preds <- initial_predictions[[model_name]]

    if (is.list(model_preds) && !is.null(names(model_preds))) {
      # Already structured by cluster_id, extract directly
      structured_predictions[[model_name]] <- if (!is.null(model_preds[[char_cluster_id]])) {
        model_preds[[char_cluster_id]]
      } else {
        "Prediction_Missing"
      }
    } else if (is.character(model_preds)) {
      # Parse text lines using shared helper (pass cluster ID for positional fallback)
      parsed <- parse_text_predictions(model_preds, c(char_cluster_id))
      structured_predictions[[model_name]] <- if (!is.null(parsed[[char_cluster_id]])) {
        parsed[[char_cluster_id]]
      } else {
        "Prediction_Missing"
      }
    }
  }

  # Create the discussion log with extracted predictions
  discussion_log <- list(
    cluster_id = char_cluster_id,
    initial_predictions = structured_predictions,
    rounds = list()
  )

  # Initialize clustering discussion log file
  get_logger()$log_discussion(char_cluster_id, "start", list(
    tissue_name = tissue_name,
    marker_genes = cluster_genes
  ))

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

    response <- tryCatch(
      get_model_response(
        prompt = first_round_prompt,
        model = model,
        api_key = api_key,
        base_urls = base_urls
      ),
      error = function(e) {
        log_warn(sprintf("Model %s failed in round 1 for cluster %s: %s",
                        model, char_cluster_id, e$message))
        paste0("Error: ", e$message)
      }
    )

    round1_responses[[model]] <- response

    # Log the model prediction to discussion file
    get_logger()$log_discussion(char_cluster_id, "prediction", list(
      model = model,
      round = 1,
      prediction = response
    ))
  }

  discussion_log$rounds[[1]] <- list(
    round_number = 1,
    responses = round1_responses
  )

  # Filter out error responses before consensus check
  valid_round1_responses <- filter_valid_responses(round1_responses, char_cluster_id)

  # Check if we have enough valid responses to continue
  if (length(valid_round1_responses) < 2) {
    log_warn(sprintf("Only %d valid responses received for cluster %s. Skipping discussion.",
                    length(valid_round1_responses), char_cluster_id))

    # Extract best available prediction or "Unknown"
    best_prediction <- if(length(valid_round1_responses) == 1) {
      response_text <- valid_round1_responses[[1]]
      if (is.character(response_text) && length(response_text) > 0) {
        cell_type_match <- regexpr("CELL TYPE:\\s*(.+)", response_text[1], ignore.case = TRUE)
        if (cell_type_match > 0) {
          trimws(sub("CELL TYPE:\\s*", "", response_text[1], ignore.case = TRUE))
        } else {
          "Unknown"
        }
      } else {
        "Unknown"
      }
    } else {
      "Unknown"
    }

    # Set a pseudo-consensus result so the caller gets a consistent structure
    discussion_log$rounds[[1]]$consensus_result <- list(
      reached = FALSE,
      consensus_proportion = 0,
      entropy = 0,
      majority_prediction = best_prediction
    )
    get_logger()$log_discussion(char_cluster_id, "consensus", discussion_log$rounds[[1]]$consensus_result)
    get_logger()$log_discussion(char_cluster_id, "end", list(
      final_result = best_prediction,
      rounds_completed = 1,
      consensus_reached = FALSE
    ))
    return(discussion_log)
  }

  # Check consensus after first round with valid responses only
  consensus_result <- check_consensus(valid_round1_responses, api_keys, controversy_threshold, entropy_threshold, consensus_check_model, base_urls)
  log_info("Consensus check completed", list(
    round = 1,
    consensus_reached = consensus_result$reached,
    consensus_proportion = consensus_result$consensus_proportion,
    entropy = consensus_result$entropy
  ))

  # Store consensus result in discussion log
  discussion_log$rounds[[1]]$consensus_result <- consensus_result
  
  # Log consensus result
  get_logger()$log_discussion(char_cluster_id, "consensus", consensus_result)

  if (consensus_result$reached && consensus_result$consensus_proportion >= controversy_threshold && consensus_result$entropy <= entropy_threshold) {
    consensus_reached <- TRUE
    # Log discussion end  
    get_logger()$log_discussion(char_cluster_id, "end", list(
      final_result = consensus_result$majority_prediction,
      rounds_completed = 1,
      consensus_reached = TRUE
    ))
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

      response <- tryCatch(
        get_model_response(
          prompt = discussion_prompt,
          model = model,
          api_key = api_key,
          base_urls = base_urls
        ),
        error = function(e) {
          log_warn(sprintf("Model %s failed in round %d for cluster %s: %s",
                          model, round, char_cluster_id, e$message))
          paste0("Error: ", e$message)
        }
      )

      round_responses[[model]] <- response

      # Log the model prediction to discussion file
      get_logger()$log_discussion(char_cluster_id, "prediction", list(
        model = model,
        round = round,
        prediction = response
      ))
    }

    discussion_log$rounds[[round]] <- list(
      round_number = round,
      responses = round_responses
    )

    # Filter out error responses before consensus check
    valid_round_responses <- filter_valid_responses(round_responses, char_cluster_id, round)
    
    # Check if we have enough valid responses to continue
    if (length(valid_round_responses) < 2) {
      log_warn(sprintf("Only %d valid responses in round %d for cluster %s. Ending discussion.", 
                      length(valid_round_responses), round, char_cluster_id))
      break  # Exit the discussion loop
    }

    # Check if consensus is reached with valid responses only
    consensus_result <- check_consensus(valid_round_responses, api_keys, controversy_threshold, entropy_threshold, consensus_check_model, base_urls)
    log_info("Consensus check completed", list(
      round = round,
      consensus_reached = consensus_result$reached,
      consensus_proportion = consensus_result$consensus_proportion,
      entropy = consensus_result$entropy
    ))

    # Store consensus result in discussion log
    discussion_log$rounds[[round]]$consensus_result <- consensus_result

    # Log consensus result
    get_logger()$log_discussion(char_cluster_id, "consensus", consensus_result)

    # Check if consensus conditions are met
    if (consensus_result$reached && consensus_result$consensus_proportion >= controversy_threshold && consensus_result$entropy <= entropy_threshold) {
      consensus_reached <- TRUE
      message(sprintf("Consensus reached in round %d with consensus proportion %.2f and entropy %.2f. Stopping discussion.",
                     round, consensus_result$consensus_proportion, consensus_result$entropy))
      break
    } else {
      message(sprintf("No strong consensus in round %d (consensus proportion: %.2f, entropy: %.2f). %s",
                     round, consensus_result$consensus_proportion, consensus_result$entropy,
                     if (round < max_rounds) "Continuing discussion..." else "Reached maximum rounds."))
    }
  }

  # Find the last round that has a consensus_result (the last round may have
  # broken early without one if valid responses were insufficient)
  final_result <- "Unknown"
  for (r in rev(seq_along(discussion_log$rounds))) {
    cr <- discussion_log$rounds[[r]]$consensus_result
    if (!is.null(cr)) {
      final_result <- cr$majority_prediction
      break
    }
  }
  
  get_logger()$log_discussion(char_cluster_id, "end", list(
    final_result = final_result,
    rounds_completed = length(discussion_log$rounds),
    consensus_reached = consensus_reached
  ))

  discussion_log
}
