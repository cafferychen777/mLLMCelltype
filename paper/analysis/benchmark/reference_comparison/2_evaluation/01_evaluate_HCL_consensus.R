library(SingleR)
library(Seurat)
library(dplyr)
library(anndata)
library(celldex)
library(Matrix)
library(HGNChelper)
library(openxlsx)
library(SingleCellExperiment)
library(scuttle)
library(scran)

# Source local ScType functions
source("analysis/benchmark/reference_comparison/2_evaluation/gene_sets_prepare.R")
source("analysis/benchmark/reference_comparison/2_evaluation/sctype_score_.R")

# Load HCL data
hcl_path <- "data/raw/HCL/HCL_Fig1_adata.h5ad"
hcl_data <- read_h5ad(hcl_path)

# Print data structure for debugging
print("Data structure:")
print(str(hcl_data))
print("Var names:")
print(names(hcl_data$var))

# Print dimensions and lengths for debugging
print("Matrix dimensions:")
print(dim(hcl_data$X))
print("Length of gene_ids:")
print(length(hcl_data$var$gene_ids))
print("Length of var_names:")
print(length(hcl_data$var_names))

# Get gene names and cell names
gene_names <- hcl_data$var_names  # Use var_names instead of gene_ids
cell_names <- hcl_data$obs_names

print("First few gene names:")
print(head(gene_names))
print("First few cell names:")
print(head(cell_names))

# Create counts matrix with correct dimensions
counts_matrix <- as(as.matrix(t(hcl_data$X)), "dgCMatrix")  # Transpose the matrix
print("Counts matrix dimensions:")
print(dim(counts_matrix))

# Set the row and column names
rownames(counts_matrix) <- gene_names
colnames(counts_matrix) <- cell_names

# Convert gene symbols to standard format
gene_symbols <- toupper(gene_names)  # Convert to uppercase

# Print some debugging information about genes
print("Number of genes in test dataset:")
print(length(gene_symbols))
print("Sample of gene names in test dataset:")
print(head(gene_symbols, 20))

print("Number of genes in reference dataset:")
ref <- HumanPrimaryCellAtlasData()
print(nrow(ref))
print("Sample of gene names in reference dataset:")
print(head(rownames(ref), 20))

print("Number of common genes:")
common_genes <- intersect(gene_symbols, rownames(ref))
print(length(common_genes))
print("Sample of common genes:")
print(head(common_genes, 20))

# Create SingleCellExperiment object
sce <- SingleCellExperiment(assays = list(counts = counts_matrix))

# Add size factors and logcounts
sce <- computeSumFactors(sce)
sce <- logNormCounts(sce)

# Run SingleR
print("Running SingleR...")
singler_results <- SingleR(
  test = sce,
  ref = ref,
  labels = ref$label.main,
  de.method = "wilcox"
)

# Get cluster-level annotations
print("Getting cluster-level annotations...")
singler_cluster_labels <- data.frame(
  cell = colnames(counts_matrix),
  cluster = hcl_data$obs$tissue,  # Use tissue instead of leiden_clusters
  singler_label = singler_results$labels
)

# Aggregate to cluster level
singler_cluster_labels <- singler_cluster_labels %>%
  group_by(cluster) %>%
  summarise(
    singler_label = names(which.max(table(singler_label)))
  )

print("SingleR cluster labels:")
print(singler_cluster_labels)

# Process ScType
print("Processing ScType...")

# Get gene sets from the local database
db_path <- "analysis/benchmark/reference_comparison/2_evaluation/ScTypeDB_full.xlsx"
tissue <- "Immune system"  # Other options: "Liver" or "Pancreas"
gs_list <- gene_sets_prepare(db_path, tissue)

# Create Seurat object
print("Creating Seurat object...")
seurat_obj <- CreateSeuratObject(counts = counts_matrix, project = "HCL")
seurat_obj$tissue <- hcl_data$obs$tissue  # Use tissue instead of leiden_clusters

# Normalize data
print("Normalizing data...")
seurat_obj <- NormalizeData(seurat_obj)

# Find variable features
print("Finding variable features...")
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)

# Scale data
print("Scaling data...")
seurat_obj <- ScaleData(seurat_obj)

# Get scaled data matrix
print("Getting scaled data...")
scaled_data <- GetAssayData(seurat_obj, slot = "scale.data")

# Run ScType scoring
print("Running ScType scoring...")
es.max <- sctype_score(scRNAseqData = scaled_data, scaled = TRUE,
                      gs = gs_list$gs_positive, gs2 = gs_list$gs_negative)

# Process results for each cluster
print("Processing cluster results...")
cL_results <- do.call("rbind", lapply(unique(hcl_data$obs$tissue), function(cl){
  es.max.cl = sort(rowSums(es.max[, seurat_obj$tissue == cl]), decreasing = TRUE)
  head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, ncells = sum(seurat_obj$tissue == cl)), 10)
}))

# Set low-confident clusters to "Unknown"
cL_results$type[as.numeric(as.character(cL_results$scores)) < cL_results$ncells/4] <- "Unknown"

# Create final comparison table
print("Creating comparison table...")
comparison_table <- data.frame(
  Cluster = unique(hcl_data$obs$tissue),
  SingleR = singler_cluster_labels$singler_label[match(unique(hcl_data$obs$tissue),
                                                     singler_cluster_labels$cluster)],
  ScType = cL_results$type[match(unique(hcl_data$obs$tissue),
                                cL_results$cluster)],
  Original_Annotation = unique(hcl_data$obs$tissue)
)

# Print results
print("Final comparison table:")
print(comparison_table)

# Save results
print("Saving results...")
write.csv(comparison_table,
          "results/benchmark/reference_comparison/2_evaluation/hcl_cell_type_comparison.csv",
          row.names = FALSE)
