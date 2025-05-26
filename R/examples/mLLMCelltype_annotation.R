library(mLLMCelltype)
library(Seurat)
library(dplyr)
library(ggplot2)
library(cowplot)
library(qs)

# Read clusters and markers
sce <- qread("/Users/apple/Downloads/sce_clusterd.qs")
filtered_markers <- qread("/Users/apple/Research/mLLMCelltype/R/examples/sce_findallmarkers_filter_markers.qs")

# Annotation by OpenRouter
consensus_results_OR <- interactive_consensus_annotation(
  input = filtered_markers,
  tissue_name = "human bonw marrow from patient with acute myeloid leukemia",
  models = c(
    "openai/gpt-4o",                       #OpenAI
    "anthropic/claude-3-7-sonnet-20250219",             #Anthropic
    "google/gemini-2.5-pro-preview-03-25", #Gemini
    "deepseek/deepseek-chat",              #Deepseek
    "qwen-max-2025-01-25",                  #qwen
    "grok-3"
  ),
  api_keys = list(
    openrouter = "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",
    qwen = "sk-4d1266fd8cac4ef7a36efe0c628d68a6",
    grok = "xai-bPJ4eF7YPDC0FefC7109W4ocFzpHIsN7yBxy1BdwQa7AB7v2oambMo3rz6baNNL8G6oA8M68EyjKzFIM"
  ),
  consensus_check_model = "anthropic/claude-3-7-sonnet-20250219"
)

# Annotation by LLM API
consensus_results <- interactive_consensus_annotation(
  input = filtered_markers,
  tissue_name = "human bonw marrow from patient with acute myeloid leukemia",
  models = c(
    "deepseek-chat",
    "qwen-max-2025-01-25",
    "grok-3",
    'gemini-2.5-pro'
  ),
  api_keys = list(
    deepseek = "sk-b9d954f8449949168dcab3b21ed1dd22",
    qwen = "sk-4d1266fd8cac4ef7a36efe0c628d68a6",
    grok = "xai-bPJ4eF7YPDC0FefC7109W4ocFzpHIsN7yBxy1BdwQa7AB7v2oambMo3rz6baNNL8G6oA8M68EyjKzFIM",
    gemini = "AIzaSyDqD_trJuYDSIYO1TWHlOCveV9qQXUJ6uE"
  ),
  consensus_check_model = "deepseek-chat"  # Use DeepSeek for consensus checking
)

# Print structure of results to understand the data
print("Available fields in consensus_results:")
print(names(consensus_results))

# Add annotations to Seurat object
# Get cell type annotations from consensus_results$final_annotations
cluster_to_celltype_map <- consensus_results$final_annotations

# Create new cell type identifier column
cell_types <- as.character(Idents(sce))
for (cluster_id in names(cluster_to_celltype_map)) {
  cell_types[cell_types == cluster_id] <- cluster_to_celltype_map[[cluster_id]]
}

# Add cell type annotations to Seurat object
sce$cell_type <- cell_types

# Add uncertainty metrics
# Extract detailed consensus results containing metrics
consensus_details <- consensus_results$initial_results$consensus_results

# Create a data frame with metrics for each cluster
uncertainty_metrics <- data.frame(
  cluster_id = names(consensus_details),
  consensus_proportion = sapply(consensus_details, function(res) res$consensus_proportion),
  entropy = sapply(consensus_details, function(res) res$entropy)
)

# Add uncertainty metrics for each cell
sce$consensus_proportion <- uncertainty_metrics$consensus_proportion[match(sce$seurat_clusters, uncertainty_metrics$cluster_id)]
sce$entropy <- uncertainty_metrics$entropy[match(sce$seurat_clusters, uncertainty_metrics$cluster_id)]

# Visualize results with SCpubr for publication-ready plots
if (!requireNamespace("SCpubr", quietly = TRUE)) {
  remotes::install_github("enblacar/SCpubr")
}
library(SCpubr)
library(viridis)  # For color palettes

# Basic UMAP visualization with default settings
pdf("pbmc_basic_annotations.pdf", width=8, height=6)
SCpubr::do_DimPlot(sample = sce,
                   group.by = "cell_type",
                   label = TRUE,
                   legend.position = "right") +
  ggtitle("mLLMCelltype Consensus Annotations")
dev.off()

# B细胞
# 嗜碱性粒细胞
# 树突状细胞
# 红细胞祖细胞
# 造血干细胞
# 造血干细胞/祖细胞
# 巨噬细胞
# 肥大细胞
# 巨核细胞
# 间质干细胞
# 有丝分裂细胞
# 单核细胞
# 中性粒细胞
# 浆细胞树突状细胞
# 增殖B细胞祖细胞
# 增殖细胞
# T细胞

# More customized visualization with enhanced styling
pdf("pbmc_custom_annotations.pdf", width=8, height=6)
SCpubr::do_DimPlot(sample = sce,
                   group.by = "cell_type",
                   label = TRUE,
                   label.box = TRUE,
                   legend.position = "right",
                   pt.size = 1.0,
                   border.size = 1,
                   font.size = 12) +
  ggtitle("mLLMCelltype Consensus Annotations") +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

# Visualize uncertainty metrics with enhanced SCpubr plots
# Get cell types and create a named color palette
cell_types <- unique(sce$cell_type)
color_palette <- viridis::viridis(length(cell_types))
names(color_palette) <- cell_types

# Cell type annotations with SCpubr
p1 <- SCpubr::do_DimPlot(sample = sce,
                         group.by = "cell_type",
                         label = TRUE,
                         legend.position = "bottom",  # Place legend at the bottom
                         pt.size = 1.0,
                         label.size = 4,  # Smaller label font size
                         label.box = TRUE,  # Add background box to labels for better readability
                         repel = TRUE,  # Make labels repel each other to avoid overlap
                         colors.use = color_palette,
                         plot.title = "Cell Type") +
  theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.3, "cm"),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Consensus proportion feature plot with SCpubr
p2 <- SCpubr::do_FeaturePlot(sample = sce,
                             features = "consensus_proportion",
                             order = TRUE,
                             pt.size = 1.0,
                             enforce_symmetry = FALSE,
                             legend.title = "Consensus",
                             plot.title = "Consensus Proportion",
                             sequential.palette = "YlGnBu",  # Yellow-Green-Blue gradient, following Nature Methods standards
                             sequential.direction = 1,  # Light to dark direction
                             min.cutoff = min(sce$consensus_proportion),  # Set minimum value
                             max.cutoff = max(sce$consensus_proportion),  # Set maximum value
                             na.value = "lightgrey") +  # Color for missing values
  theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Shannon entropy feature plot with SCpubr
p3 <- SCpubr::do_FeaturePlot(sample = sce,
                             features = "entropy",
                             order = TRUE,
                             pt.size = 1.0,
                             enforce_symmetry = FALSE,
                             legend.title = "Entropy",
                             plot.title = "Shannon Entropy",
                             sequential.palette = "OrRd",  # Orange-Red gradient, following Nature Methods standards
                             sequential.direction = -1,  # Dark to light direction (reversed)
                             min.cutoff = min(sce$entropy),  # Set minimum value
                             max.cutoff = max(sce$entropy),  # Set maximum value
                             na.value = "lightgrey") +  # Color for missing values
  theme(plot.title = element_text(hjust = 0.5, margin = margin(b = 15, t = 10)),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))

# Combine plots with equal widths
pdf("sce_uncertainty_metrics.pdf", width=18, height=7)
combined_plot <- cowplot::plot_grid(p1, p2, p3, ncol = 3, rel_widths = c(1.2, 1.2, 1.2))
print(combined_plot)
dev.off()
