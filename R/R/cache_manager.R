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
    
    #' @description Generate cache key from input parameters (improved version)
    #' @param input Input data
    #' @param models Models used
    #' @param cluster_id Cluster ID
    #' @return Cache key string
    generate_key = function(input, models, cluster_id) {
      # Extract genes using a standardized approach
      genes <- private$extract_genes_standardized(input, cluster_id)
      
      # Create standardized components with input context for empty genes
      genes_hash <- private$create_genes_hash(genes, input, cluster_id)
      models_hash <- private$create_models_hash(models)
      cluster_hash <- private$create_cluster_hash(cluster_id)
      
      # Combine into final key with version prefix
      key <- paste("v", self$cache_version, genes_hash, models_hash, cluster_hash, sep = "_")
      
      return(key)
    },
    
    #' @description Save results to cache
    #' @param key Cache key
    #' @param data Data to cache
    save_to_cache = function(key, data) {
      # Ensure cache directory exists
      if (!dir.exists(self$cache_dir)) {
        dir.create(self$cache_dir, recursive = TRUE)
      }
      
      # Create cache file path
      cache_file <- file.path(self$cache_dir, paste0(key, ".rds"))
      
      # Try to save with error handling
      tryCatch({
        saveRDS(data, cache_file)
        get_logger()$log_cache_operation("store", key, file.size(cache_file))
      }, error = function(e) {
        get_logger()$log_cache_operation("store_failed", key, NULL)
        warning(paste("Failed to save cache file:", e$message))
        # Try to create parent directories if they don't exist
        dir.create(dirname(cache_file), recursive = TRUE, showWarnings = FALSE)
        # Try again
        tryCatch({
          saveRDS(data, cache_file)
          get_logger()$log_cache_operation("store", key, file.size(cache_file))
        }, error = function(e2) {
          get_logger()$log_cache_operation("store_failed", key, NULL)
          warning(paste("Second attempt to save cache file failed:", e2$message))
        })
      })
    },
    
    #' @description Load results from cache
    #' @param key Cache key
    #' @return Cached data if exists, NULL otherwise
    load_from_cache = function(key) {
      cache_file <- file.path(self$cache_dir, paste0(key, ".rds"))
      if (file.exists(cache_file)) {
        get_logger()$log_cache_operation("hit", key, file.size(cache_file))
        return(readRDS(cache_file))
      }
      get_logger()$log_cache_operation("miss", key)
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
      get_logger()$log_cache_operation("clear", "all_cache", length(cache_files))
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
  ),
  
  private = list(
    #' Extract genes from input in a standardized way
    #' @param input Input data (list or data.frame)
    #' @param cluster_id Cluster ID
    #' @return Character vector of gene names
    extract_genes_standardized = function(input, cluster_id) {
      tryCatch({
        if (is.list(input) && !is.data.frame(input)) {
          # Handle list input
          cluster_key <- as.character(cluster_id)
          if (cluster_key %in% names(input) && 
              is.list(input[[cluster_key]]) && 
              "genes" %in% names(input[[cluster_key]])) {
            genes <- input[[cluster_key]]$genes
            return(sort(unique(as.character(genes))))
          }
        } else if (is.data.frame(input)) {
          # Handle data frame input
          if (all(c("cluster", "gene") %in% names(input))) {
            # Convert cluster_id to character for consistent matching
            char_cluster_id <- as.character(cluster_id)
            
            # Filter for the specific cluster
            cluster_mask <- input$cluster == char_cluster_id
            
            # If no match and cluster_id could be numeric, try numeric matching
            if (!any(cluster_mask) && !is.na(suppressWarnings(as.numeric(char_cluster_id)))) {
              cluster_mask <- input$cluster == as.numeric(char_cluster_id)
            }
            
            # Extract genes, optionally filtering by avg_log2FC if available
            cluster_data <- input[cluster_mask, ]
            if ("avg_log2FC" %in% names(cluster_data)) {
              cluster_data <- cluster_data[cluster_data$avg_log2FC > 0, ]
            }
            
            if (nrow(cluster_data) > 0) {
              genes <- cluster_data$gene
              return(sort(unique(as.character(genes))))
            }
          }
        }
        
        # Fallback: return empty character vector
        return(character(0))
        
      }, error = function(e) {
        get_logger()$warn(paste("Error extracting genes:", e$message))
        return(character(0))
      })
    },
    
    #' Create stable hash from genes list
    #' @param genes Character vector of gene names
    #' @param input Original input data (for context when genes is empty)
    #' @param cluster_id Cluster ID (for context when genes is empty)
    #' @return Hash string
    create_genes_hash = function(genes, input = NULL, cluster_id = NULL) {
      if (length(genes) == 0) {
        # For empty gene lists, create hash based on input data characteristics
        # This provides more differentiation than just "no_genes"
        context_info <- list()
        
        if (!is.null(input) && !is.null(cluster_id)) {
          if (is.data.frame(input)) {
            # Include information about the input data structure
            context_info$total_genes <- nrow(input)
            context_info$total_clusters <- length(unique(input$cluster))
            context_info$cluster_id <- as.character(cluster_id)
            
            # Include a sample of all genes to differentiate datasets
            if ("gene" %in% names(input)) {
              all_genes <- sort(unique(as.character(input$gene)))
              # Use first and last few genes as signature
              gene_signature <- c(
                head(all_genes, 3),
                tail(all_genes, 3)
              )
              context_info$gene_signature <- paste(gene_signature, collapse = "_")
            }
          } else if (is.list(input)) {
            # For list input, use the structure
            context_info$list_length <- length(input)
            context_info$cluster_id <- as.character(cluster_id)
            context_info$available_clusters <- paste(sort(names(input)), collapse = "_")
          }
        }
        
        if (length(context_info) > 0) {
          context_hash <- digest::digest(context_info, algo = "xxhash64")
          return(paste0("empty_", substr(context_hash, 1, 12)))
        } else {
          return("empty_unknown")
        }
      }
      
      # Sort and deduplicate for consistency
      genes_clean <- sort(unique(as.character(genes)))
      
      # Use a fast hash algorithm
      digest::digest(genes_clean, algo = "xxhash64")
    },
    
    #' Create stable hash from models list
    #' @param models Character vector of model names
    #' @return Hash string
    create_models_hash = function(models) {
      if (length(models) == 0) {
        return("no_models")
      }
      
      # Sort for consistency and create abbreviated hash
      models_sorted <- sort(unique(as.character(models)))
      digest::digest(models_sorted, algo = "xxhash64")
    },
    
    #' Create stable hash from cluster ID
    #' @param cluster_id Cluster identifier
    #' @return Hash string
    create_cluster_hash = function(cluster_id) {
      # Always convert to character for consistency
      cluster_str <- as.character(cluster_id)
      
      # For short cluster IDs, return directly; for long ones, hash
      if (nchar(cluster_str) <= 8) {
        return(paste0("c", cluster_str))
      } else {
        return(paste0("c", substr(digest::digest(cluster_str, algo = "xxhash64"), 1, 8)))
      }
    }
  )
)
