library(ggplot2)
library(patchwork)
library(magick)

# 准备PDF文件路径
pdf_a <- "manuscript/figures/panel_a.pdf"
pdf_b <- "manuscript/figures/panel_b.pdf"
pdf_c <- "manuscript/figures/panel_c.pdf"

# 检查PDF文件是否存在
if (!file.exists(pdf_a) || !file.exists(pdf_b) || !file.exists(pdf_c)) {
  stop("一个或多个PDF文件不存在，请确保所有PDF文件都已生成")
}

# 直接从PDF读取图像
figure2a <- image_read_pdf(pdf_a, density=300)
figure2b <- image_read_pdf(pdf_b, density=300)
figure2c <- image_read_pdf(pdf_c, density=300)

# 创建自定义主题，保留少量边距
small_margin_theme <- theme_void() +
  theme(
    plot.margin = margin(1, 2, 1, 2, "mm"),  # 上、右、下、左边距各2mm
    plot.title = element_text(face = "bold", size = 16, hjust = 0, vjust = 1, margin = margin(t = 0, r = 0, b = 0, l = 2, unit = "pt"))
  )

# 使用自定义主题创建图形
p1 <- ggplot() +
  annotation_custom(grid::rasterGrob(as.raster(figure2a), interpolate=TRUE,
                                    width=unit(0.98, "npc"), height=unit(0.98, "npc")),
                   xmin=-1, xmax=1, ymin=-1, ymax=1) +
  scale_x_continuous(expand = c(0.01, 0.01), limits = c(-1, 1)) +
  scale_y_continuous(expand = c(0.01, 0.01), limits = c(-1, 1)) +
  small_margin_theme +
  labs(title = "a")

p3 <- ggplot() +
  annotation_custom(grid::rasterGrob(as.raster(figure2b), interpolate=TRUE,
                                    width=unit(0.98, "npc"), height=unit(0.98, "npc")),
                   xmin=-1, xmax=1, ymin=-1, ymax=1) +
  scale_x_continuous(expand = c(0.01, 0.01), limits = c(-1, 1)) +
  scale_y_continuous(expand = c(0.01, 0.01), limits = c(-1, 1)) +
  small_margin_theme +
  labs(title = "b")

p4 <- ggplot() +
  annotation_custom(grid::rasterGrob(as.raster(figure2c), interpolate=TRUE,
                                    width=unit(0.98, "npc"), height=unit(0.98, "npc")),
                   xmin=-1, xmax=1, ymin=-1, ymax=1) +
  scale_x_continuous(expand = c(0.01, 0.01), limits = c(-1, 1)) +
  scale_y_continuous(expand = c(0.01, 0.01), limits = c(-1, 1)) +
  small_margin_theme +
  labs(title = "c")

# 布局设计
layout <- c(
  area(t = 1, l = 1, b = 2, r = 4),  # figure2a 占据第1-2行，所有列
  area(t = 3, l = 1, b = 3, r = 4),  # figure2b 占据第3行，所有列
  area(t = 4, l = 1, b = 4, r = 4)   # figure2c 占据第4行，所有列
)

# 组合图像，设置适当的面板间距
combined_plot <- p1 + p3 + p4 +
  plot_layout(design = layout) &
  theme(
    plot.margin = margin(1, 1, 1, 1, "mm"),
    panel.spacing = unit(1, "mm")
  )

# 直接保存为PDF，不再生成PNG
# 注意: 输出为 figure4.pdf (Benchmarks)
ggsave("manuscript/figures/figure4.pdf",
       combined_plot, width = 12, height = 12, dpi = 300)

# 打印完成信息
cat("图像已成功合并并保存为figure4.pdf (Benchmarks)，分辨率为300dpi\n")
