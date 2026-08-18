library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(viridis)

# Set the theme to match Nature Methods style
theme_nature <- function(base_size = 8) {
  theme_classic(base_size = base_size) +
    theme(
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.text = element_text(color = "black", size = 8),
      axis.title = element_text(color = "black", size = 9),
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9),
      plot.title = element_text(size = 10, face = "bold"),
      strip.background = element_blank(),
      strip.text = element_text(size = 9, face = "bold")
    )
}

# Generate sample data
set.seed(42)
n_clusters <- 50

# Data for panel a
global_acc <- runif(n_clusters, 0.75, 0.85)
annotation_data <- data.frame(
  cluster_id = 1:n_clusters,
  global_accuracy = global_acc,
  hierarchical_accuracy = pmin(1, global_acc + runif(n_clusters, 0.03, 0.07))
)

# Create panel a - Scatter plot comparison
panel_a <- ggplot(annotation_data, aes(x = global_accuracy, y = hierarchical_accuracy)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  geom_point(size = 2, alpha = 0.7, color = "#2171B5") +
  scale_x_continuous(limits = c(0.7, 1), breaks = seq(0.7, 1, 0.1)) +
  scale_y_continuous(limits = c(0.7, 1), breaks = seq(0.7, 1, 0.1)) +
  labs(
    x = "Global Marker Gene Approach Accuracy",
    y = "Hierarchical Strategy Accuracy",
    title = "a"
  ) +
  theme_nature()

# Data for panel b
accuracy_improvement <- data.frame(
  annotation_level = factor(c("Level 2", "Level 3", "Level 4"),
                          levels = c("Level 2", "Level 3", "Level 4")),
  global_accuracy = c(0.88, 0.85, 0.84),
  hierarchical_accuracy = c(0.92, 0.89, 0.87)
)

# Print original data
print("Original data:")
print(accuracy_improvement)

# Transform data
accuracy_improvement <- accuracy_improvement %>%
  pivot_longer(
    cols = c(global_accuracy, hierarchical_accuracy),
    names_to = "method",
    values_to = "accuracy"
  ) %>%
  mutate(
    method = factor(method,
                   levels = c("global_accuracy", "hierarchical_accuracy"),
                   labels = c("Global Marker Gene", "Hierarchical Strategy"))
  )

# Print transformed data
print("\nTransformed data:")
print(accuracy_improvement)

# Print summary statistics
print("\nSummary statistics:")
print(summary(accuracy_improvement$accuracy))

# Create panel b - Bar plot
panel_b <- ggplot(accuracy_improvement,
                 aes(x = annotation_level, y = accuracy, fill = method)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c("#6BAED6", "#2171B5")) +
  scale_y_continuous(breaks = seq(0.82, 0.94, 0.02)) +
  coord_cartesian(ylim = c(0.82, 0.94)) +
  labs(
    x = "Annotation Level",
    y = "Annotation Accuracy",
    fill = "Method",
    title = "b"
  ) +
  theme_nature() +
  theme(
    legend.position = "bottom",
    legend.box.margin = margin(-10, 0, 0, 0)
  )

# Combine panels
combined_plot <- panel_a + panel_b +
  plot_layout(widths = c(1, 1.2))

# Save the plot
ggsave(
  "manuscript/figures/supplementary_figure7.pdf",
  combined_plot,
  width = 180,
  height = 90,
  units = "mm",
  dpi = 300
)

# Also save as PNG for quick preview
ggsave(
  "manuscript/figures/supplementary_figure7.png",
  combined_plot,
  width = 180,
  height = 90,
  units = "mm",
  dpi = 300
)
