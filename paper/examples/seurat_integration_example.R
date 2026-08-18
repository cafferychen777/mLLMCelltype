# Seurat Integration Example for mLLMCelltype
# This script demonstrates how to use mLLMCelltype with Seurat for cell type annotation
#
# Usage:
#   1. Set your API keys as environment variables or in a .env file
#   2. Run this script: Rscript seurat_integration_example.R

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Check and install required packages
required_packages <- c("Seurat", "dplyr")
new_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
if(length(new_packages) > 0) {
  install.packages(new_packages)
}

# Load required packages
library(mLLMCelltype)
library(Seurat)
library(dplyr)

# =============================================================================
# Configuration - Set your API keys
# =============================================================================
# Option 1: Environment variables (recommended)
# export OPENAI_API_KEY="your-key"
# export ANTHROPIC_API_KEY="your-key"

# Option 2: Load from .env file in current directory
if (file.exists(".env")) {
  if (requireNamespace("dotenv", quietly = TRUE)) {
    dotenv::load_dot_env(".env")
  }
}

# Get API keys from environment
openai_key <- Sys.getenv("OPENAI_API_KEY")
anthropic_key <- Sys.getenv("ANTHROPIC_API_KEY")

# Check if API keys are available
if (openai_key == "" && anthropic_key == "") {
  stop("No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.")
}

# Create results directory
results_dir <- "./results"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# Download and load PBMC dataset
# =============================================================================
pbmc_data_url <- "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
pbmc_data_dir <- "pbmc3k_filtered_gene_bc_matrices"

# Create a directory for the data if it doesn't exist
if (!dir.exists(pbmc_data_dir)) {
  dir.create(pbmc_data_dir)

  # Download and extract the data
  temp_file <- tempfile(fileext = ".tar.gz")
  download.file(pbmc_data_url, temp_file)
  untar(temp_file, exdir = pbmc_data_dir)
  unlink(temp_file)
}

# Load the PBMC dataset
pbmc_data <- Read10X(data.dir = file.path(pbmc_data_dir, "filtered_gene_bc_matrices/hg19/"))

# Create Seurat object
pbmc <- CreateSeuratObject(counts = pbmc_data, project = "pbmc3k", min.cells = 3, min.features = 200)

# =============================================================================
# Quality control and preprocessing
# =============================================================================
# Add mitochondrial gene percentage information
pbmc[['percent.mt']] <- PercentageFeatureSet(pbmc, pattern = '^MT-')

# Visualize QC metrics
print("Plotting QC metrics...")
pdf(file.path(results_dir, "pbmc_qc.pdf"))
p1 <- VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
print(p1)
dev.off()

# Filter cells
pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

# =============================================================================
# Standard Seurat workflow
# =============================================================================
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# Scale data and run PCA
pbmc <- ScaleData(pbmc, features = rownames(pbmc))
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# Cluster the cells
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5)

# Run UMAP
pbmc <- RunUMAP(pbmc, dims = 1:10)

# View clusters
print("Plotting UMAP with clusters...")
pdf(file.path(results_dir, "pbmc_clusters.pdf"))
DimPlot(pbmc, reduction = "umap", label = TRUE)
dev.off()

# =============================================================================
# Find marker genes
# =============================================================================
print("Finding marker genes for each cluster...")
pbmc_markers <- FindAllMarkers(pbmc,
                             only.pos = TRUE,
                             min.pct = 0.25,
                             logfc.threshold = 0.25)

# Print top markers for each cluster
top_markers <- pbmc_markers %>%
  group_by(cluster) %>%
  top_n(5, avg_log2FC)
print("Top markers for each cluster:")
print(top_markers)

# =============================================================================
# Run mLLMCelltype annotation
# =============================================================================
print("Running mLLMCelltype annotation...")

# Build API keys list based on available keys
api_keys <- list()
models <- c()

if (anthropic_key != "") {
  api_keys$anthropic <- anthropic_key
  models <- c(models, "claude-sonnet-4-5-20250929")
}
if (openai_key != "") {
  api_keys$openai <- openai_key
  models <- c(models, "gpt-4o")
}

consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  models = models,
  api_keys = api_keys,
  top_gene_count = 10
)

# =============================================================================
# Add annotations to Seurat object
# =============================================================================
print("Adding annotations to Seurat object...")

# Get final annotation results
cluster_annotations <- consensus_results$final_annotations

# Make sure cluster_annotations is a simple character vector
if(is.list(cluster_annotations)) {
  cluster_annotations <- unlist(cluster_annotations)
}

# Create a named vector for mapping (0-based indexing for Seurat)
cluster_map <- cluster_annotations
names(cluster_map) <- as.character(0:(length(cluster_map)-1))

# Add annotations to cells
current_idents <- as.character(Idents(pbmc))
cell_annotations <- cluster_map[current_idents]
pbmc$mLLMCelltype <- cell_annotations

# =============================================================================
# Visualize results
# =============================================================================
print("Plotting cell type annotations...")
pdf(file.path(results_dir, "pbmc_annotations.pdf"))
DimPlot(pbmc, group.by = "mLLMCelltype", label = TRUE, repel = TRUE) +
  ggtitle("mLLMCelltype Annotations")
dev.off()

# Save the annotated Seurat object
saveRDS(pbmc, file.path(results_dir, "pbmc_annotated.rds"))

# Print final annotations
print("Final cell type annotations:")
print(consensus_results$final_annotations)

print("Analysis complete. Results saved to the results directory.")
