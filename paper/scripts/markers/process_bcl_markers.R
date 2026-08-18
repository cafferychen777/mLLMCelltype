# Load required libraries
library(Seurat)
library(SeuratDisk)
library(dplyr)

# Set paths
project_root <- here::here()
input_file <- file.path(project_root, "data/raw/BCL.h5ad")
output_file_top10 <- file.path(project_root, "data/reference/BCL_markers.csv")
output_file_top20 <- file.path(project_root, "data/reference/BCL_markers_top20.csv")
output_file_top30 <- file.path(project_root, "data/reference/BCL_markers_top30.csv")

# Convert h5ad to h5seurat
Convert(input_file, dest = "h5seurat", overwrite = TRUE)

# Load the converted file
bcl_seurat <- LoadH5Seurat("BCL.h5seurat")

# Normalize data if not already normalized
if(!"RNA" %in% names(bcl_seurat@assays)) {
    bcl_seurat <- CreateSeuratObject(counts = bcl_seurat@assays$RNA@counts)
}
bcl_seurat <- NormalizeData(bcl_seurat, normalization.method = "LogNormalize", scale.factor = 10000)

# Find markers
markers <- FindAllMarkers(bcl_seurat,
                         only.pos = TRUE,
                         min.pct = 0.1,
                         logfc.threshold = 0.25,
                         test.use = "wilcox",
                         return.thresh = 0.05)

# Function to process markers and save to file
save_top_markers <- function(markers, n_top, output_file) {
    # Sort markers by p-value and then by logfc for ties
    markers <- markers %>%
        group_by(cluster) %>%
        arrange(p_val, desc(avg_log2FC)) %>%
        slice_head(n = n_top) %>%
        ungroup()

    # Create output format
    result_lines <- c("cell_type,gene")

    # Process each cluster
    for(clust in unique(markers$cluster)) {
        cluster_markers <- markers %>%
            filter(cluster == clust) %>%
            pull(gene)
        result_lines <- c(result_lines,
                         paste(clust, paste(cluster_markers, collapse = ","), sep = ","))
    }

    # Write to file
    writeLines(result_lines, output_file)

    # Print summary
    cat(sprintf("\nSaved top %d markers for %d cell types to %s\n",
                n_top, length(unique(markers$cluster)), output_file))
}

# Save different versions
save_top_markers(markers, 10, output_file_top10)
save_top_markers(markers, 20, output_file_top20)
save_top_markers(markers, 30, output_file_top30)

# Print data summary
print("\nData Summary:")
print(paste("Number of cells:", ncol(bcl_seurat)))
print(paste("Number of genes:", nrow(bcl_seurat)))
print("\nCell type distribution:")
print(table(bcl_seurat$cell_type))

# Clean up
file.remove("BCL.h5seurat")
