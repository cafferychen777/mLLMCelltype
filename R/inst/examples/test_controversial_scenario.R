# Script to test interactive_consensus_annotation with controversial clusters

# Load required packages
library(devtools)
library(ggplot2)

# Load dotenv package
if (!requireNamespace("dotenv", quietly = TRUE)) {
  install.packages("dotenv")
}
library(dotenv)

# Set working directory to the package root
setwd("/Users/apple/Research/LLMCelltype/code/R")

# Load .env file
project_root <- "/Users/apple/Research/LLMCelltype"
dotenv::load_dot_env(file.path(project_root, ".env"))

# Verify API keys are loaded
cat("QWEN_API_KEY loaded:", nchar(Sys.getenv("QWEN_API_KEY")) > 0, "\n")
cat("ANTHROPIC_API_KEY loaded:", nchar(Sys.getenv("ANTHROPIC_API_KEY")) > 0, "\n")
cat("OPENAI_API_KEY loaded:", nchar(Sys.getenv("OPENAI_API_KEY")) > 0, "\n")
cat("GEMINI_API_KEY loaded:", nchar(Sys.getenv("GEMINI_API_KEY")) > 0, "\n")

# Load API keys from environment variables
api_keys <- list(
  # Model names as keys
  "claude-3-5-sonnet-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
  "gemini-1.5-pro" = Sys.getenv("GEMINI_API_KEY"),
  "qwen-max-2025-01-25" = Sys.getenv("QWEN_API_KEY"),
  "gpt-4o" = Sys.getenv("OPENAI_API_KEY"),
  
  # Provider names as keys (needed by some functions)
  "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
  "gemini" = Sys.getenv("GEMINI_API_KEY"),
  "qwen" = Sys.getenv("QWEN_API_KEY"),
  "openai" = Sys.getenv("OPENAI_API_KEY")
)

# Create ambiguous mock data that will likely create controversial annotations
create_controversial_mock_data <- function() {
  # Create a list of genes for each cluster with some ambiguity
  data <- list(
    # Cluster 1: Mix of B cell and plasma cell markers (controversial)
    "1" = list(
      genes = c("CD19", "CD20", "CD79A", "CD79B", "PAX5", "SDC1", "PRDM1", "XBP1", "IRF4", "JCHAIN", 
               "MZB1", "IGHA1", "IGHG1", "IGHM", "CD38"),
      description = "B cell/Plasma cell markers"
    ),
    
    # Cluster 2: Mix of CD4+ and CD8+ T cell markers (controversial)
    "2" = list(
      genes = c("CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "FOXP3", "IL2RA", "CTLA4", "GZMA", 
               "GZMB", "PRF1", "CCR7", "IL7R", "CD27"),
      description = "Mixed T cell markers"
    ),
    
    # Cluster 3: Mix of classical and non-classical monocyte markers (controversial)
    "3" = list(
      genes = c("CD14", "CD16", "FCGR3A", "FCGR3B", "CSF1R", "CX3CR1", "CCR2", "CD163", "MS4A7", "S100A8", 
               "S100A9", "LYZ", "VCAN", "HLA-DRA", "HLA-DRB1"),
      description = "Mixed monocyte markers"
    ),
    
    # Cluster 4: Mix of NK cell and innate lymphoid cell markers (controversial)
    "4" = list(
      genes = c("NCAM1", "NKG7", "KLRD1", "KLRB1", "KLRC1", "IL7R", "GATA3", "RORA", "RORC", "IL23R", 
               "IL1R1", "IL22", "KIT", "ICOS", "PTGDR2"),
      description = "NK/ILC markers"
    ),
    
    # Cluster 5: Mix of myeloid DC and plasmacytoid DC markers (controversial)
    "5" = list(
      genes = c("CD1C", "CLEC10A", "CLEC4C", "LILRA4", "IL3RA", "TCF4", "IRF8", "IRF7", "TLR7", "TLR9", 
               "SIGLEC6", "AXL", "FCER1A", "ITGAX", "ITGAM"),
      description = "Mixed DC markers"
    ),
    
    # Cluster 6: Mix of megakaryocyte and platelet markers (controversial)
    "6" = list(
      genes = c("PPBP", "PF4", "ITGA2B", "GP1BA", "GP9", "VWF", "SELP", "CD9", "MPL", "FLI1", 
               "GATA1", "NFE2", "RUNX1", "TAL1", "PLEK"),
      description = "Megakaryocyte/Platelet markers"
    ),
    
    # Cluster 7: Mix of erythroid progenitor and mature erythrocyte markers (controversial)
    "7" = list(
      genes = c("HBB", "HBA1", "HBA2", "GYPA", "GYPB", "TFRC", "CD71", "KIT", "EPOR", "GATA1", 
               "KLF1", "TAL1", "SLC4A1", "EPB42", "ANK1"),
      description = "Erythroid lineage markers"
    )
  )
  return(data)
}

# Create a temporary directory for logs
temp_log_dir <- tempfile("test_logs")
dir.create(temp_log_dir)
cat("Created temporary log directory:", temp_log_dir, "\n")

# Create a temporary directory for cache
temp_cache_dir <- tempfile("test_cache")
dir.create(temp_cache_dir)
cat("Created temporary cache directory:", temp_cache_dir, "\n")

# Load the package
cat("\nLoading package...\n")
load_all(".")

# Run the test with controversial data
cat("\nTesting interactive_consensus_annotation with controversial data...\n")

# Create controversial mock data
mock_data <- create_controversial_mock_data()

# Set tissue name
tissue_name <- "Human PBMC"

# Set models to use (use multiple models for a more realistic test)
test_models <- c(
  "claude-3-5-sonnet-latest",  # Anthropic
  "gemini-1.5-pro",           # Google
  "qwen-max-2025-01-25",      # Qwen
  "gpt-4o"                    # OpenAI
)

# Set a higher controversy threshold to force discussion
controversy_threshold <- 0.9  # This high threshold will likely trigger discussions

# Run the refactored function with controversial data
result <- tryCatch({
  # Use the refactored function
  interactive_consensus_annotation(
    input = mock_data,
    tissue_name = tissue_name,
    models = test_models,
    api_keys = api_keys,
    top_gene_count = 15,  # Use all genes
    controversy_threshold = controversy_threshold,
    entropy_threshold = 0.5,  # Lower entropy threshold to catch more controversies
    max_discussion_rounds = 2,
    log_dir = temp_log_dir,
    cache_dir = temp_cache_dir,
    use_cache = TRUE
  )
}, error = function(e) {
  cat("Error occurred during annotation:", e$message, "\n")
  return(e)
})

# Check if the result is an error
if (inherits(result, "error")) {
  cat("\nTest failed with error!\n")
} else {
  cat("\nTest succeeded!\n")
  
  # Print the results
  cat("\nFinal annotations:\n")
  print(result$final_annotations)
  
  # Print controversial clusters
  if (length(result$controversial_clusters) > 0) {
    cat("\nControversial clusters that required discussion:\n")
    print(result$controversial_clusters)
    
    # Print discussion logs for controversial clusters
    cat("\nDiscussion logs for controversial clusters:\n")
    for (cluster_id in result$controversial_clusters) {
      cat("\nCluster", cluster_id, "discussion:\n")
      if (!is.null(result$discussion_logs[[cluster_id]])) {
        print(result$discussion_logs[[cluster_id]])
      } else {
        cat("No discussion log available\n")
      }
    }
  } else {
    cat("\nNo controversial clusters were identified despite ambiguous data.\n")
  }
  
  # Check if log files were created
  log_files <- list.files(temp_log_dir)
  cat("\nLog files created:", length(log_files), "\n")
  if (length(log_files) > 0) {
    print(log_files)
  }
  
  # Create a visualization of the results
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    # Create a data frame for plotting
    plot_data <- data.frame(
      Cluster = names(result$final_annotations),
      CellType = unlist(result$final_annotations),
      stringsAsFactors = FALSE
    )
    
    # Create a bar plot
    p <- ggplot(plot_data, aes(x = Cluster, y = 1, fill = CellType)) +
      geom_bar(stat = "identity") +
      theme_minimal() +
      labs(title = "Cell Type Annotations (After Discussion)", x = "Cluster", y = "") +
      theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
      coord_flip() +
      scale_fill_brewer(palette = "Set3")
    
    # Save the plot
    plot_file <- file.path(temp_log_dir, "controversial_annotations_plot.png")
    ggsave(plot_file, p, width = 10, height = 6)
    cat("\nCreated visualization at:", plot_file, "\n")
  }
}

cat("\nDone!\n")
