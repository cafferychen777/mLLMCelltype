# mLLMCelltype Visualization Script - Cell Types Grouped by Major Categories
#
# This script demonstrates advanced visualization of cell type annotations,
# including grouping by major cell lineages with customized UMAP plots.
#
# Usage:
#   1. Set your API keys as environment variables
#   2. Run this script: Rscript cell_type_visualization.R

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# =============================================================================
# Configuration - Set your API keys
# =============================================================================
# Load from .env file if available
if (file.exists(".env")) {
  if (requireNamespace("dotenv", quietly = TRUE)) {
    dotenv::load_dot_env(".env")
  }
}

# Load required packages
library(mLLMCelltype)
library(Seurat)
library(dplyr)
library(ggplot2)

# Create results directory
results_dir <- "./results"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# Download and process PBMC dataset
# =============================================================================
pbmc_data_url <- "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
pbmc_data_dir <- "pbmc3k_filtered_gene_bc_matrices"

# If data directory doesn't exist, download and extract data
if (!dir.exists(pbmc_data_dir)) {
  dir.create(pbmc_data_dir)
  temp_file <- tempfile(fileext = ".tar.gz")
  download.file(pbmc_data_url, temp_file)
  untar(temp_file, exdir = pbmc_data_dir)
  unlink(temp_file)
}

# Load PBMC dataset
pbmc_data <- Read10X(data.dir = file.path(pbmc_data_dir, "filtered_gene_bc_matrices/hg19/"))

# Create Seurat object
pbmc <- CreateSeuratObject(counts = pbmc_data, project = "pbmc3k", min.cells = 3, min.features = 200)

# Add mitochondrial gene percentage information
pbmc[['percent.mt']] <- PercentageFeatureSet(pbmc, pattern = '^MT-')

# Filter cells
pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

# Standard processing workflow
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
pbmc <- ScaleData(pbmc, features = rownames(pbmc))
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5)
pbmc <- RunUMAP(pbmc, dims = 1:10)

# Find marker genes for each cluster
pbmc_markers <- FindAllMarkers(pbmc,
                             only.pos = TRUE,
                             min.pct = 0.25,
                             logfc.threshold = 0.25)

# =============================================================================
# Run mLLMCelltype annotation
# =============================================================================
# Get API keys
anthropic_key <- Sys.getenv("ANTHROPIC_API_KEY")
openai_key <- Sys.getenv("OPENAI_API_KEY")
google_key <- Sys.getenv("GEMINI_API_KEY")
qwen_key <- Sys.getenv("QWEN_API_KEY")

# Build models and api_keys lists based on available keys
models <- c()
api_keys <- list()

if (anthropic_key != "") {
  models <- c(models, "claude-sonnet-4-5-20250929")
  api_keys$anthropic <- anthropic_key
}
if (openai_key != "") {
  models <- c(models, "gpt-4o")
  api_keys$openai <- openai_key
}
if (google_key != "") {
  models <- c(models, "gemini-2.0-flash-exp")
  api_keys$gemini <- google_key
}
if (qwen_key != "") {
  models <- c(models, "qwen-max-2025-01-25")
  api_keys$qwen <- qwen_key
}

if (length(models) == 0) {
  stop("No API keys found. Please set at least one API key environment variable.")
}

# Run mLLMCelltype annotation using multiple LLM models
consensus_results <- interactive_consensus_annotation(
  input = pbmc_markers,
  tissue_name = "human PBMC",
  models = models,
  api_keys = api_keys,
  top_gene_count = 10
)

# Get final annotation results
final_annotations <- unlist(consensus_results$final_annotations)

# Create mapping from cluster ID to cell type (0-based for Seurat)
cluster_to_celltype_map <- setNames(
  final_annotations,
  as.numeric(names(consensus_results$final_annotations)) - 1
)

# Assign cell types to each cell
current_clusters <- as.numeric(as.character(Idents(pbmc)))
cell_types <- sapply(current_clusters, function(cl) cluster_to_celltype_map[as.character(cl)])
pbmc$mLLMCelltype <- cell_types

# =============================================================================
# Create cell type groupings
# =============================================================================
unique_cell_types <- unique(pbmc$mLLMCelltype)

# Dynamically create cell type groupings based on actual annotation results
cell_type_groups <- list(
  "T Cells" = grep("T cells|T cell|CD4|CD8|Cytotoxic|Naive", unique_cell_types, value = TRUE, ignore.case = TRUE),
  "B Cells" = grep("B cells|B cell", unique_cell_types, value = TRUE, ignore.case = TRUE),
  "Myeloid Cells" = grep("Monocyte|Macrophage|Dendritic|DC", unique_cell_types, value = TRUE, ignore.case = TRUE),
  "NK Cells" = grep("NK|Natural Killer", unique_cell_types, value = TRUE, ignore.case = TRUE),
  "Other" = grep("Platelet", unique_cell_types, value = TRUE, ignore.case = TRUE)
)

# Add unassigned cell types to Other group
all_grouped_cells <- unlist(cell_type_groups)
unassigned_cells <- setdiff(unique_cell_types, all_grouped_cells)
if (length(unassigned_cells) > 0) {
  cell_type_groups[["Other"]] <- c(cell_type_groups[["Other"]], unassigned_cells)
}

# Function to get cell group
get_cell_group <- function(cell_type, group_list) {
  for (group_name in names(group_list)) {
    if (cell_type %in% group_list[[group_name]]) {
      return(group_name)
    }
  }
  return("Other")
}

# Assign major category to each cell type
cell_type_to_group <- sapply(unique_cell_types, function(ct) get_cell_group(ct, cell_type_groups))
names(cell_type_to_group) <- unique_cell_types

# Add major category to Seurat object
cell_group_vector <- cell_type_to_group[pbmc$mLLMCelltype]
names(cell_group_vector) <- colnames(pbmc)
pbmc[['cell_group']] <- cell_group_vector

# =============================================================================
# Create visualization
# =============================================================================
# Set custom color scheme
custom_colors <- c(
  "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
  "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
)

# Get UMAP coordinates and cell type information
umap_data <- Embeddings(pbmc, reduction = 'umap')
umap_df <- as.data.frame(umap_data)
umap_df$celltype <- pbmc$mLLMCelltype
umap_df$cell_group <- pbmc$cell_group

# Create basic UMAP plot with cell type annotations
p_basic <- DimPlot(pbmc,
         reduction = "umap",
         group.by = "mLLMCelltype",
         label = TRUE,
         repel = TRUE,
         cols = custom_colors,
         pt.size = 0.7) +
  theme_void() +
  theme(
    legend.text = element_text(size = 12),
    legend.position = "right",
    plot.margin = margin(t = 20, r = 20, b = 20, l = 20, unit = "pt")
  ) +
  guides(color = guide_legend(override.aes = list(size = 5), title = NULL)) +
  ggtitle("mLLMCelltype Cell Type Annotations")

# Save basic plot
pdf(file.path(results_dir, "pbmc_annotations_basic.pdf"), width = 10, height = 8)
print(p_basic)
dev.off()

# Create plot grouped by major categories
p_grouped <- DimPlot(pbmc,
         reduction = "umap",
         group.by = "cell_group",
         label = TRUE,
         repel = TRUE,
         pt.size = 0.7) +
  theme_void() +
  theme(
    legend.text = element_text(size = 12),
    legend.position = "right",
    plot.margin = margin(t = 20, r = 20, b = 20, l = 20, unit = "pt")
  ) +
  guides(color = guide_legend(override.aes = list(size = 5), title = NULL)) +
  ggtitle("Cell Types by Major Category")

# Save grouped plot
pdf(file.path(results_dir, "pbmc_annotations_grouped.pdf"), width = 10, height = 8)
print(p_grouped)
dev.off()

# Save high-resolution PNG
png(file.path(results_dir, "pbmc_annotations_highres.png"), width = 3000, height = 2400, res = 300)
print(p_basic)
dev.off()

# =============================================================================
# Save results
# =============================================================================
saveRDS(pbmc, file.path(results_dir, "pbmc_annotated_grouped.rds"))
saveRDS(consensus_results, file.path(results_dir, "consensus_results.rds"))

# Print summary
print("Cell type annotation results:")
print(table(pbmc$mLLMCelltype))

print("\nCell major category grouping results:")
print(table(pbmc$cell_group))

message("Analysis complete, results saved in the results directory")
