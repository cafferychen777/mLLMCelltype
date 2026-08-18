# Load required libraries
library(Seurat)
library(dplyr)
library(reticulate)
library(tidyr)

# Set paths
project_root <- "."
input_file <- file.path(project_root, "data/raw/BCL.h5ad")
output_file <- file.path(project_root, "data/reference/BCL_markers.csv")

# Install and import required Python packages
use_condaenv("r-reticulate")
py_install(c("anndata", "scanpy"), pip = TRUE)

# Import Python modules
anndata <- import("anndata")
sc <- import("scanpy")

# Read the h5ad file
print("Reading h5ad file...")
adata <- anndata$read_h5ad(input_file)

# Convert to Seurat object
print("Converting to Seurat object...")
counts <- t(adata$X$toarray())
gene_names <- adata$var_names$to_list()
cell_names <- adata$obs_names$to_list()
metadata <- adata$obs$to_dict()
cell_types <- adata$obs$refined_clusters$to_list()  # Using refined_clusters instead of cell_type

# Create Seurat object
rownames(counts) <- gene_names
colnames(counts) <- cell_names
bcl_seurat <- CreateSeuratObject(counts = counts)
bcl_seurat$cell_type <- cell_types

# Normalize data
print("Normalizing data...")
bcl_seurat <- NormalizeData(bcl_seurat, normalization.method = "LogNormalize", scale.factor = 10000)

# Find markers
print("Finding marker genes...")
markers <- FindAllMarkers(bcl_seurat,
                         only.pos = TRUE,
                         min.pct = 0.1,
                         logfc.threshold = 0.25,
                         test.use = "wilcox",
                         return.thresh = 0.05)

# Process markers and save to CSV
print("Processing markers...")
markers_list <- markers %>%
  group_by(cluster) %>%
  arrange(p_val, desc(avg_log2FC)) %>%
  slice_head(n = 10) %>%
  summarise(genes = paste(gene, collapse = ",")) %>%
  mutate(genes = paste(cluster, genes, sep = ","))

# Create output directory if it doesn't exist
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

# Save as CSV
print("Saving markers to CSV...")
writeLines(c("cluster,gene", markers_list$genes), output_file)

# Print data summary
print("\nData Summary:")
print(paste("Number of cells:", ncol(bcl_seurat)))
print(paste("Number of genes:", nrow(bcl_seurat)))
print("\nCluster distribution:")
print(table(bcl_seurat$cell_type))

print(paste("\nMarkers saved to:", output_file))
