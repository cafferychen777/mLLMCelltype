#' Logger class for cell type annotation discussions
#' @importFrom R6 R6Class
#' @export
DiscussionLogger <- R6::R6Class("DiscussionLogger",
  public = list(
    #' @field log_dir Directory for storing log files
    log_dir = NULL,
    
    #' @field current_log Current log file handle
    current_log = NULL,
    
    #' @field session_id Unique identifier for the current session
    session_id = NULL,
    
    #' @description
    #' Initialize a new logger
    #' @param base_dir Base directory for logs
    initialize = function(base_dir = "logs") {
      self$log_dir <- base_dir
      if (!dir.exists(self$log_dir)) {
        dir.create(self$log_dir, recursive = TRUE)
      }
      self$session_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
      private$create_session_dir()
    },
    
    #' @description
    #' Start logging a new cluster discussion
    #' @param cluster_id Cluster identifier
    #' @param tissue_name Tissue name
    #' @param marker_genes List of marker genes
    start_cluster_discussion = function(cluster_id, tissue_name = NULL, marker_genes) {
      private$close_current_log()
      
      log_file <- file.path(private$session_dir, sprintf("cluster_%s_discussion.log", cluster_id))
      self$current_log <- file(log_file, "w")
      
      # Log initial information
      self$log_entry("DISCUSSION_START", list(
        cluster_id = cluster_id,
        tissue_name = tissue_name,
        marker_genes = marker_genes,
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      ))
    },
    
    #' @description
    #' Log a discussion entry
    #' @param event_type Type of event
    #' @param content Content to log
    log_entry = function(event_type, content) {
      if (is.null(self$current_log)) {
        warning("No active log file. Call start_cluster_discussion first.")
        return(invisible(NULL))
      }
      
      # Convert content to character if it's not already
      if (is.list(content)) {
        content <- lapply(content, function(x) {
          if (is.null(x)) return("NULL")
          if (is.list(x)) return(jsonlite::toJSON(x, auto_unbox = TRUE))
          toString(x)
        })
      }
      
      entry <- list(
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        event_type = event_type,
        content = content
      )
      
      # Write entry to log file
      writeLines(jsonlite::toJSON(entry, auto_unbox = TRUE, pretty = TRUE),
                self$current_log)
      writeLines(",", self$current_log)  # Add separator for multiple entries
    },
    
    #' @description
    #' Log a model's prediction
    #' @param model_name Name of the model
    #' @param round_number Discussion round number
    #' @param prediction Model's prediction and reasoning
    log_prediction = function(model_name, round_number, prediction) {
      self$log_entry("MODEL_PREDICTION", list(
        model = model_name,
        round = round_number,
        prediction = prediction
      ))
    },
    
    #' @description
    #' Log consensus check results
    #' @param round_number Round number
    #' @param reached_consensus Whether consensus was reached
    #' @param consensus_proportion Proportion of models supporting the majority prediction
    #' @param entropy Shannon entropy of the predictions (optional)
    log_consensus_check = function(round_number, reached_consensus, consensus_proportion, entropy = NULL) {
      self$log_entry("CONSENSUS_CHECK", list(
        round = round_number,
        consensus_reached = reached_consensus,
        consensus_proportion = consensus_proportion,
        entropy = entropy
      ))
    },
    
    #' @description
    #' Log final consensus result
    #' @param final_cell_type Final determined cell type
    #' @param summary Summary of the discussion
    log_final_consensus = function(final_cell_type, summary) {
      self$log_entry("FINAL_CONSENSUS", list(
        cell_type = final_cell_type,
        summary = summary
      ))
    },
    
    #' @description
    #' End current cluster discussion and close log file
    end_cluster_discussion = function() {
      if (!is.null(self$current_log)) {
        self$log_entry("DISCUSSION_END", list(
          timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
        ))
        private$close_current_log()
      }
    }
  ),
  
  private = list(
    session_dir = NULL,
    
    create_session_dir = function() {
      private$session_dir <- file.path(self$log_dir, self$session_id)
      if (!dir.exists(private$session_dir)) {
        dir.create(private$session_dir, recursive = TRUE)
      }
    },
    
    close_current_log = function() {
      if (!is.null(self$current_log)) {
        close(self$current_log)
        self$current_log <- NULL
      }
    }
  )
)
