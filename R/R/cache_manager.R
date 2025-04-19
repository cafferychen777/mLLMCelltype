#' Cache Manager Class
#' @description Manages caching of consensus analysis results
#' @importFrom R6 R6Class
#' @importFrom digest digest
#' @export
CacheManager <- R6::R6Class(
  "CacheManager",
  
  public = list(
    #' @field cache_dir Directory to store cache files
    cache_dir = NULL,
    
    #' @field cache_version Current cache version
    cache_version = "1.0",
    
    #' @description Initialize cache manager
    #' @param cache_dir Directory to store cache files
    initialize = function(cache_dir = "consensus_cache") {
      self$cache_dir <- cache_dir
      if (!dir.exists(cache_dir)) {
        dir.create(cache_dir, recursive = TRUE)
      }
    },
    
    #' @description Generate cache key from input parameters
    #' @param input Input data
    #' @param models Models used
    #' @param cluster_id Cluster ID
    #' @return Cache key string
    generate_key = function(input, models, cluster_id) {
      # Add more robust type checking and error handling
      tryCatch({
        # For list input, use gene names
        if (is.list(input) && !is.data.frame(input)) {
          # Check if input[[cluster_id]] is a list
          if (is.list(input[[cluster_id]]) && "genes" %in% names(input[[cluster_id]])) {
            genes <- input[[cluster_id]]$genes
            input_hash <- digest::digest(sort(as.character(genes)))
          } else {
            # If input[[cluster_id]] is not a list or doesn't have a 'genes' element
            message("Warning: Expected input[[cluster_id]] to be a list with a 'genes' element")
            # Use cluster_id as an alternative hash source
            input_hash <- digest::digest(paste("cluster", cluster_id, sep="_"))
          }
        } else if (is.data.frame(input)) {
          # For data frame input, use cluster's genes
          if (all(c("cluster", "avg_log2FC", "gene") %in% names(input))) {
            # Use original cluster ID, no conversion from 1-based to 0-based indexing
            cluster_data <- input[input$cluster == as.numeric(cluster_id) & input$avg_log2FC > 0, ]
            if (nrow(cluster_data) > 0) {
              input_hash <- digest::digest(sort(as.character(cluster_data$gene)))
            } else {
              message("Warning: No genes found for cluster ", cluster_id)
              input_hash <- digest::digest(paste("empty_cluster", cluster_id, sep="_"))
            }
          } else {
            message("Warning: Input data frame missing required columns")
            input_hash <- digest::digest(paste("invalid_input", cluster_id, sep="_"))
          }
        } else {
          # If input is neither a list nor a data frame
          message("Warning: Input is neither a list nor a data frame")
          input_hash <- digest::digest(paste("unknown_input", cluster_id, sep="_"))
        }
        
        # Create key from input hash and models
        key <- paste(input_hash, paste(sort(models), collapse = "_"), cluster_id, sep = "_")
        
        # Add version information to the key
        key <- paste(key, self$cache_version, sep = "_v")
        return(key)
      }, error = function(e) {
        # Catch all errors and return a key based on the error message
        message("Error in generate_key: ", e$message)
        error_hash <- digest::digest(paste("error", cluster_id, e$message, sep="_"))
        key <- paste(error_hash, paste(sort(models), collapse = "_"), cluster_id, sep = "_")
        return(key)
      })
    },
    
    #' @description Save results to cache
    #' @param key Cache key
    #' @param data Data to cache
    save_to_cache = function(key, data) {
      cache_file <- file.path(self$cache_dir, paste0(key, ".rds"))
      saveRDS(data, cache_file)
    },
    
    #' @description Load results from cache
    #' @param key Cache key
    #' @return Cached data if exists, NULL otherwise
    load_from_cache = function(key) {
      cache_file <- file.path(self$cache_dir, paste0(key, ".rds"))
      if (file.exists(cache_file)) {
        return(readRDS(cache_file))
      }
      return(NULL)
    },
    
    #' @description Check if results exist in cache
    #' @param key Cache key
    #' @return TRUE if cached results exist
    has_cache = function(key) {
      cache_file <- file.path(self$cache_dir, paste0(key, ".rds"))
      return(file.exists(cache_file))
    },
    
    #' @description Get cache statistics
    #' @return A list with cache statistics
    get_cache_stats = function() {
      if (!dir.exists(self$cache_dir)) {
        return(list(
          cache_exists = FALSE,
          cache_count = 0,
          cache_size_mb = 0
        ))
      }
      
      cache_files <- list.files(self$cache_dir, pattern = "\\.rds$", full.names = TRUE)
      cache_sizes <- file.size(cache_files)
      
      return(list(
        cache_exists = TRUE,
        cache_count = length(cache_files),
        cache_size_mb = sum(cache_sizes) / (1024 * 1024),
        cache_files = cache_files
      ))
    },
    
    #' @description Clear all cache
    #' @param confirm Boolean, if TRUE, will clear cache without confirmation
    clear_cache = function(confirm = FALSE) {
      if (!dir.exists(self$cache_dir)) {
        message("Cache directory does not exist.")
        return(invisible(NULL))
      }
      
      cache_files <- list.files(self$cache_dir, pattern = "\\.rds$", full.names = TRUE)
      
      if (length(cache_files) == 0) {
        message("No cache files to clear.")
        return(invisible(NULL))
      }
      
      if (!confirm) {
        message(sprintf("This will delete %d cache files. Set confirm=TRUE to proceed.", length(cache_files)))
        return(invisible(NULL))
      }
      
      unlink(cache_files)
      message(sprintf("Cleared %d cache files.", length(cache_files)))
      
      return(invisible(NULL))
    },
    
    #' @description Validate cache content
    #' @param key Cache key
    #' @return TRUE if cache is valid, FALSE otherwise
    validate_cache = function(key) {
      if (!self$has_cache(key)) {
        return(FALSE)
      }
      
      tryCatch({
        cache_data <- self$load_from_cache(key)
        
        # Check cache data structure
        if (!is.list(cache_data)) {
          return(FALSE)
        }
        
        # Check required fields
        required_fields <- c("annotation", "discussion_log")
        if (!all(required_fields %in% names(cache_data))) {
          return(FALSE)
        }
        
        # Check discussion_log structure
        if (!is.list(cache_data$discussion_log) || 
            !("rounds" %in% names(cache_data$discussion_log))) {
          return(FALSE)
        }
        
        return(TRUE)
      }, error = function(e) {
        message("Error validating cache: ", e$message)
        return(FALSE)
      })
    }
  )
)
