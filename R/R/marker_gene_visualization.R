#' Create Enhanced Marker Gene Bubble Plot
#'
#' Creates a publication-ready bubble plot showing marker gene expression across cell types.
#' The plot displays both the percentage of cells expressing each gene (bubble size) and
#' the average expression level (color intensity).
#'
#' @param seurat_obj A Seurat object containing the single-cell data
#' @param markers_df A data frame containing marker genes (output from FindAllMarkers)
#' @param consensus_results Results from interactive_consensus_annotation function
#' @param top_n Number of top marker genes to display per cell type (default: 5)
#' @param title Plot title (default: "Marker Gene Expression by Cell Type")
#' @param color_palette Color palette for expression levels (default: "plasma")
#'
#' @return A list containing the ggplot object and the underlying data
#'
#' @examples
#' \dontrun{
#' # Run consensus annotation first
#' consensus_results <- interactive_consensus_annotation(
#'   input = markers_df,
#'   tissue_name = "human PBMC",
#'   models = c("anthropic/claude-3.5-sonnet", "openai/gpt-4o"),
#'   api_keys = list(openrouter = "your_api_key")
#' )
#' 
#' # Create bubble plot
#' bubble_plot <- create_marker_bubble_plot(
#'   seurat_obj = pbmc_data,
#'   markers_df = markers_df,
#'   consensus_results = consensus_results,
#'   top_n = 5
#' )
#' 
#' # Display the plot
#' print(bubble_plot$plot)
#' }
#'
#' @import ggplot2
#' @import dplyr
#' @import viridis
#' @import Seurat
#' @export
create_marker_bubble_plot <- function(seurat_obj, markers_df, consensus_results, 
                                     top_n = 5, 
                                     title = "Marker Gene Expression by Cell Type",
                                     color_palette = "plasma") {
  
  # Check required packages
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for this function.")
  }
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Package 'dplyr' is required for this function.")
  }
  if (!requireNamespace("viridis", quietly = TRUE)) {
    stop("Package 'viridis' is required for this function.")
  }
  
  # Get final annotations
  final_annotations <- consensus_results$final_annotations
  
  # Get top marker genes for each cluster
  top_markers <- markers_df %>%
    dplyr::group_by(cluster) %>%
    dplyr::top_n(top_n, avg_log2FC) %>%
    dplyr::arrange(cluster, desc(avg_log2FC))
  
  # Create a mapping from cluster to cell type
  # Handle the case where final_annotations is a list
  if (is.list(final_annotations)) {
    cluster_to_celltype <- unlist(final_annotations)
  } else {
    cluster_to_celltype <- setNames(final_annotations, names(final_annotations))
  }
  
  # Add cell type information to markers
  top_markers$cell_type <- cluster_to_celltype[as.character(top_markers$cluster)]
  
  # Calculate average expression and percentage expressed for each gene in each cell type
  bubble_data <- top_markers %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      avg_exp = mean(Seurat::GetAssayData(seurat_obj, layer = "data")[gene, 
                                  Seurat::WhichCells(seurat_obj, idents = cluster)], na.rm = TRUE),
      pct_exp = sum(Seurat::GetAssayData(seurat_obj, layer = "data")[gene, 
                                 Seurat::WhichCells(seurat_obj, idents = cluster)] > 0) / 
                length(Seurat::WhichCells(seurat_obj, idents = cluster)) * 100,
      log2fc = avg_log2FC
    ) %>%
    dplyr::ungroup()
  
  # Create enhanced bubble plot with better aesthetics
  p <- ggplot2::ggplot(bubble_data, ggplot2::aes(x = reorder(cell_type, cluster), 
                                                 y = reorder(gene, avg_exp))) +
    ggplot2::geom_point(ggplot2::aes(size = pct_exp, color = avg_exp), alpha = 0.8) +
    ggplot2::scale_size_continuous(
      name = "% Expressed", 
      range = c(2, 12),
      breaks = c(25, 50, 75, 100),
      guide = ggplot2::guide_legend(override.aes = list(alpha = 1))
    ) +
    viridis::scale_color_viridis_c(
      name = "Avg Expression\n(log2)", 
      option = color_palette,
      trans = "log10",
      labels = function(x) round(x, 2)
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      axis.text.y = ggplot2::element_text(size = 10, face = "italic"),
      axis.title = ggplot2::element_text(size = 14, face = "bold"),
      plot.title = ggplot2::element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = ggplot2::element_text(size = 12, hjust = 0.5, color = "grey40"),
      legend.title = ggplot2::element_text(size = 12, face = "bold"),
      legend.text = ggplot2::element_text(size = 10),
      panel.grid.major = ggplot2::element_line(color = "grey95", linewidth = 0.5),
      panel.grid.minor = ggplot2::element_blank(),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      plot.background = ggplot2::element_rect(fill = "white", color = NA)
    ) +
    ggplot2::labs(
      title = title,
      subtitle = "Bubble size represents percentage of cells expressing the gene\nColor intensity shows average expression level",
      x = "Cell Type",
      y = "Marker Genes"
    )
  
  return(list(plot = p, data = bubble_data))
}

#' Create Enhanced Marker Gene Heatmap
#'
#' Creates a publication-ready heatmap showing marker gene expression across cell types.
#' The heatmap displays scaled expression values with hierarchical clustering of genes.
#'
#' @param seurat_obj A Seurat object containing the single-cell data
#' @param markers_df A data frame containing marker genes (output from FindAllMarkers)
#' @param consensus_results Results from interactive_consensus_annotation function
#' @param top_n Number of top marker genes to display per cell type (default: 5)
#' @param title Plot title (default: "Marker Gene Expression Heatmap")
#' @param color_palette Color palette for the heatmap (default: viridis::plasma)
#' @param cluster_genes Whether to cluster genes hierarchically (default: TRUE)
#' @param scale_method Scaling method: "row", "column", or "none" (default: "row")
#' @param width Image width in pixels (default: 4000)
#' @param height Image height in pixels (default: 3000)
#' @param fontsize Base font size for the heatmap (default: 20)
#' @param cellwidth Width of each cell in the heatmap (default: 100)
#' @param cellheight Height of each cell in the heatmap (default: 50)
#' @param margins Margins around the heatmap as c(bottom, right) (default: c(20, 20))
#'
#' @return A matrix containing the expression data used for the heatmap
#'
#' @examples
#' \dontrun{
#' # Create heatmap with default settings (optimized for complete display)
#' heatmap_matrix <- create_marker_heatmap(
#'   seurat_obj = pbmc_data,
#'   markers_df = markers_df,
#'   consensus_results = consensus_results,
#'   top_n = 5
#' )
#'
#' # Create heatmap with custom settings
#' heatmap_matrix <- create_marker_heatmap(
#'   seurat_obj = pbmc_data,
#'   markers_df = markers_df,
#'   consensus_results = consensus_results,
#'   top_n = 3,
#'   title = "Top 3 Marker Genes",
#'   fontsize = 16,
#'   cellwidth = 80,
#'   cellheight = 40,
#'   cluster_genes = FALSE
#' )
#' }
#'
#' @import pheatmap
#' @import viridis
#' @import dplyr
#' @import Seurat
#' @export
create_marker_heatmap <- function(seurat_obj, markers_df, consensus_results,
                                 top_n = 5,
                                 title = "Marker Gene Expression Heatmap",
                                 color_palette = NULL,
                                 cluster_genes = TRUE,
                                 scale_method = "row",
                                 width = 4000,
                                 height = 3000,
                                 fontsize = 20,
                                 cellwidth = 100,
                                 cellheight = 50,
                                 margins = c(20, 20)) {

  # Check required packages
  if (!requireNamespace("pheatmap", quietly = TRUE)) {
    stop("Package 'pheatmap' is required for this function.")
  }
  if (!requireNamespace("viridis", quietly = TRUE)) {
    stop("Package 'viridis' is required for this function.")
  }

  # Set default color palette
  if (is.null(color_palette)) {
    color_palette <- viridis::plasma(100)
  }

  # Get final annotations
  final_annotations <- consensus_results$final_annotations

  # Get top marker genes for each cluster
  top_markers <- markers_df %>%
    dplyr::group_by(cluster) %>%
    dplyr::top_n(top_n, avg_log2FC) %>%
    dplyr::arrange(cluster, desc(avg_log2FC))

  # Create a mapping from cluster to cell type
  # Handle the case where final_annotations is a list
  if (is.list(final_annotations)) {
    cluster_to_celltype <- unlist(final_annotations)
  } else {
    cluster_to_celltype <- setNames(final_annotations, names(final_annotations))
  }

  # Get unique genes and cell types
  genes <- unique(top_markers$gene)
  if (is.list(final_annotations)) {
    cell_types <- unique(unlist(final_annotations))
  } else {
    cell_types <- unique(final_annotations)
  }

  # Create expression matrix
  expr_matrix <- matrix(0, nrow = length(genes), ncol = length(cell_types))
  rownames(expr_matrix) <- genes
  colnames(expr_matrix) <- cell_types

  # Fill the matrix with average expression values
  for (i in 1:nrow(top_markers)) {
    gene <- top_markers$gene[i]
    cluster <- top_markers$cluster[i]
    cell_type <- cluster_to_celltype[as.character(cluster)]

    # Calculate average expression for this gene in this cluster
    cells_in_cluster <- Seurat::WhichCells(seurat_obj, idents = cluster)
    if (length(cells_in_cluster) > 0) {
      avg_exp <- mean(Seurat::GetAssayData(seurat_obj, layer = "data")[gene, cells_in_cluster], na.rm = TRUE)
      expr_matrix[gene, cell_type] <- avg_exp
    }
  }

  # Simplify cell type names to avoid display issues
  colnames(expr_matrix) <- gsub("[^A-Za-z0-9_]", "_", colnames(expr_matrix))
  colnames(expr_matrix) <- gsub("__+", "_", colnames(expr_matrix))  # Remove multiple underscores
  colnames(expr_matrix) <- gsub("^_|_$", "", colnames(expr_matrix))  # Remove leading/trailing underscores

  # Create enhanced heatmap with optimized settings for complete display
  pheatmap::pheatmap(
    expr_matrix,
    cluster_rows = cluster_genes,
    cluster_cols = FALSE,
    scale = scale_method,
    color = color_palette,
    main = title,
    fontsize = fontsize,        # Use parameter value
    fontsize_row = fontsize - 2, # Slightly smaller for gene names
    fontsize_col = fontsize,     # Use parameter value for cell types
    cellwidth = cellwidth,       # Use parameter value
    cellheight = cellheight,     # Use parameter value
    border_color = "black",      # Visible borders for clarity
    show_rownames = TRUE,
    show_colnames = TRUE,
    angle_col = 45,              # 45-degree angled column names to prevent overlap
    treeheight_row = if(cluster_genes) 50 else 0,
    treeheight_col = 0,
    legend = TRUE,
    legend_breaks = c(-2, -1, 0, 1, 2),
    legend_labels = c("Low", "", "Medium", "", "High"),
    margins = margins            # Use parameter value for margins
  )

  return(expr_matrix)
}
