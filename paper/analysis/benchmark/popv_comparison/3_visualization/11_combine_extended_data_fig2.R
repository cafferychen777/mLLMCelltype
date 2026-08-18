library(patchwork)
library(ggplot2)
library(magick)

# Set working directory to the project root
setwd(".")

# Input files
ref_umap <- "results/figures/extended_figure2/HLCA_level3_reference_umap.pdf"
llm_umap <- "results/figures/extended_figure2/HLCA_level3_llmcelltype_umap.pdf"

# Create the output directory if it doesn't exist
dir.create("results/figures/extended_figure2", showWarnings = FALSE, recursive = TRUE)

# Output file
output_file <- "results/figures/extended_figure2/extended_data_fig2.pdf"

# Read PDFs with higher density
ref_img <- image_read_pdf(ref_umap, density=300)
llm_img <- image_read_pdf(llm_umap, density=300)

# Create plots with better image handling
p1 <- ggplot() +
  annotation_custom(grid::rasterGrob(as.raster(ref_img), interpolate=TRUE), xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) +
  theme_void() +
  theme(plot.margin = margin(0, 0, 0, 0, "pt"))

p2 <- ggplot() +
  annotation_custom(grid::rasterGrob(as.raster(llm_img), interpolate=TRUE), xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) +
  theme_void() +
  theme(plot.margin = margin(0, 0, 0, 0, "pt"))

# Combine plots using patchwork with automatic tags
combined_plot <- p1 + p2 +
  plot_layout(ncol = 2, widths = c(1, 1)) +
  plot_annotation(
    tag_levels = "a",
    theme = theme(
      plot.tag = element_text(face = "bold", size = 24),  # 增大标签字体
      plot.tag.position = c(0.1, 0.95),  # 调整标签位置，离边缘远一点
      plot.margin = margin(0, 0, -10, 0, "mm")  # 上边距20mm，其他边距5mm
    )
  )

# Save the combined plot with better dimensions
ggsave(output_file, combined_plot,
       width = 10, height = 5,  # 保持宽高比例，但减小总体尺寸
       device = cairo_pdf,
       dpi = 300)

print(paste("Combined figure saved to:", output_file))
