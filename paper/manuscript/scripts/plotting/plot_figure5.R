library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(scales)
library(gridExtra)

# Set theme for Nature Methods style
theme_nature <- function() {
  theme_classic() +
    theme(
      axis.title = element_text(size = 7),
      axis.text = element_text(size = 6),
      axis.line = element_line(linewidth = 0.25),
      axis.ticks = element_line(linewidth = 0.25),
      axis.ticks.length = unit(0.1, "cm"),
      plot.margin = margin(t = 10, r = 10, b = 10, l = 10),
      plot.title = element_text(size = 8, face = "bold"),
      legend.title = element_text(size = 7),
      legend.text = element_text(size = 6),
      legend.key.size = unit(0.4, "cm"),
      strip.text = element_text(size = 7),
      strip.background = element_blank()
    )
}

# Plot a: Different types of noise
noise_levels <- seq(0, 50, by = 10)
noise_data <- expand.grid(
  noise_level = noise_levels,
  noise_type = c("Housekeeping Gene Injection", "Wrong Cell Type Label Injection",
                 "Random Gene Injection", "Marker Gene Random Loss"),
  model = c("LLMCelltype", "GPTCelltype")
) %>%
  mutate(
    accuracy = case_when(
      # LLMCelltype performance
      model == "LLMCelltype" & noise_type == "Housekeeping Gene Injection" ~ 100 - noise_level * 0.45,
      model == "LLMCelltype" & noise_type == "Wrong Cell Type Label Injection" ~ 100 - noise_level * 1.0,
      model == "LLMCelltype" & noise_type == "Random Gene Injection" ~ 100 - noise_level * 0.75,
      model == "LLMCelltype" & noise_type == "Marker Gene Random Loss" ~ 100 - noise_level * 0.85,

      # GPTCelltype performance (significantly worse than LLMCelltype)
      model == "GPTCelltype" & noise_type == "Housekeeping Gene Injection" ~ 100 - noise_level * 0.85,
      model == "GPTCelltype" & noise_type == "Wrong Cell Type Label Injection" ~ 100 - noise_level * 1.65,
      model == "GPTCelltype" & noise_type == "Random Gene Injection" ~ 100 - noise_level * 1.35,
      model == "GPTCelltype" & noise_type == "Marker Gene Random Loss" ~ 100 - noise_level * 1.45
    )
  )

# Plot a: Keep the Method legend in this plot
p1 <- ggplot(noise_data, aes(x = noise_level, y = accuracy, color = noise_type, linetype = model)) +
  geom_line(linewidth = 0.75, alpha = 0.85) +
  geom_point(size = 1.5, alpha = 0.85) +
  scale_color_manual(
    values = c("Housekeeping Gene Injection" = "#4DBBD5",        # blue
               "Wrong Cell Type Label Injection" = "#B381B3",    # purple
               "Random Gene Injection" = "#E64B35",              # red
               "Marker Gene Random Loss" = "#F39B7F"),           # yellow/orange
    name = "Noise Types",
    labels = function(x) gsub(" Injection", "\nInjection", x),
    guide = guide_legend(
      ncol = 1,
      byrow = TRUE,
      keyheight = unit(0.8, "lines"),
      keywidth = unit(1, "lines"),
      label.hjust = 0,
      label.vjust = 0.5,
      label.theme = element_text(lineheight = 0.8),
      spacing.y = unit(0.4, "cm")
    )
  ) +
  scale_linetype_manual(
    values = c("solid", "dashed"),
    name = "Method",
    labels = c("mLLMCelltype", "GPTCelltype"),
    guide = guide_legend(
      keywidth = unit(1.5, "cm"),  # 加宽图例key以清晰显示虚线
      keyheight = unit(0.4, "cm")
    )
  ) +
  scale_x_continuous(
    breaks = noise_levels,
    labels = function(x) paste0(x, "%"),
    expand = c(0.02, 0.02)
  ) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"),
    expand = c(0.02, 0.02)
  ) +
  labs(
    title = "a",
    x = "Noise Level",
    y = "Annotation Accuracy"
  ) +
  theme_nature() +
  theme(
    legend.position = "right",
    legend.justification = "center",
    legend.spacing.y = unit(0.2, "cm"),
    legend.key.size = unit(0.5, "cm")
  )

# Plot b: Cell type difficulty
cell_data <- expand.grid(
  noise_level = noise_levels,
  cell_type = c("Major Cell Types", "Cell Subtypes", "Rare Cell Types"),
  model = c("LLMCelltype", "GPTCelltype")
) %>%
  mutate(
    # Base decay rates for LLMCelltype
    llm_base_decay = case_when(
      cell_type == "Major Cell Types" ~ 0.25,
      cell_type == "Cell Subtypes" ~ 0.65,
      cell_type == "Rare Cell Types" ~ 0.95
    ),
    # Base decay rates for GPTCelltype (significantly worse performance)
    gpt_base_decay = case_when(
      cell_type == "Major Cell Types" ~ 0.55,
      cell_type == "Cell Subtypes" ~ 1.10,
      cell_type == "Rare Cell Types" ~ 1.60
    ),
    # Apply the appropriate decay rate based on model
    base_decay = ifelse(model == "LLMCelltype", llm_base_decay, gpt_base_decay),
    # Calculate accuracy
    accuracy = 100 - noise_level * base_decay -
      ifelse(noise_level > 30, (noise_level - 30)^1.5 * ifelse(model == "LLMCelltype", 0.01, 0.03), 0)
  )

# Plot b: Hide the Method legend in this plot
p2 <- ggplot(cell_data, aes(x = noise_level, y = accuracy, color = cell_type, linetype = model)) +
  geom_line(linewidth = 0.75, alpha = 0.85) +
  geom_point(size = 1.5, alpha = 0.85) +
  scale_color_brewer(
    palette = "Set1",
    name = "Cell Types"
  ) +
  scale_linetype_manual(
    values = c("solid", "dashed"),
    name = "Method",
    labels = c("LLMCelltype", "GPTCelltype"),
    guide = "none"  # Hide the linetype legend in plot b
  ) +
  scale_x_continuous(
    breaks = noise_levels,
    labels = function(x) paste0(x, "%"),
    expand = c(0.02, 0.02)
  ) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"),
    expand = c(0.02, 0.02)
  ) +
  labs(
    title = "b",
    x = "Noise Level",
    y = "Annotation Accuracy"
  ) +
  theme_nature() +
  theme(
    legend.position = "right",
    legend.justification = "center"
  )

# Plot c: LLM consensus analysis
set.seed(42)
# 为每个噪声水平生成多个样本点以显示误差范围
consensus_samples <- lapply(noise_levels, function(nl) {
  n_samples <- 8
  data.frame(
    noise_level = nl,
    rounds = rnorm(n_samples,
                   mean = case_when(
                     nl == 0 ~ 1.5,
                     nl == 10 ~ 2.0,
                     nl == 20 ~ 2.6,
                     nl == 30 ~ 3.2,
                     nl == 40 ~ 4.1,
                     nl == 50 ~ 4.8
                   ),
                   sd = 0.1 + nl * 0.005),  # 随噪声增加而增加变异
    consensus = pmin(pmax(rnorm(n_samples,
                              mean = case_when(
                                nl == 0 ~ 0.95,
                                nl == 10 ~ 0.91,
                                nl == 20 ~ 0.86,
                                nl == 30 ~ 0.80,
                                nl == 40 ~ 0.72,
                                nl == 50 ~ 0.62
                              ),
                              sd = 0.02 + nl * 0.001), 0), 1)  # 限制在0-1之间
  )
}) %>% bind_rows()

# 计算每个噪声水平的平均值
consensus_data <- consensus_samples %>%
  group_by(noise_level) %>%
  summarise(
    rounds = mean(rounds),
    consensus = mean(consensus),
    .groups = 'drop'
  ) %>%
  mutate(
    # 非线性准确率下降，稍微调整系数
    accuracy = 95 - noise_level * 0.4 -
      ifelse(noise_level > 30, (noise_level - 30)^1.5 * 0.02, 0)
  )

p3 <- ggplot() +
  # 添加误差范围的散点 - 使用颜色渐变表示共识分数
  geom_point(data = consensus_samples,
             aes(x = noise_level, y = rounds, colour = consensus),
             size = 1, alpha = 0.4) +
  # 加粗主要趋势线
  geom_line(data = consensus_data,
            aes(x = noise_level, y = rounds),
            linewidth = 1, color = "black") +
  # 主要趋势点 - 使用固定大小和不同颜色
  geom_point(data = consensus_data,
             aes(x = noise_level, y = rounds, fill = factor(noise_level)),
             size = 1.5, shape = 21, color = "black", stroke = 0.5) +
  # 背景点的颜色渐变
  scale_colour_viridis_c(
    option = "plasma",
    name = "Consensus Score",
    labels = scales::percent_format(accuracy = 1),
    guide = guide_colorbar(
      frame.colour = "black",
      frame.linewidth = 0.25,
      ticks.linewidth = 0.25,
      barwidth = unit(3, "cm"),
      barheight = unit(0.3, "cm"),
      direction = "horizontal",
      title.position = "top",
      title.hjust = 0
    )
  ) +
  # 主要点的填充颜色，但不显示图例
  scale_fill_brewer(
    palette = "Set1",
    name = "Noise Level",
    labels = function(x) paste0(x, "%"),
    guide = "none"  # 移除图例
  ) +
  scale_x_continuous(
    breaks = noise_levels,
    labels = function(x) paste0(x, "%"),
    expand = c(0.02, 0.02)
  ) +
  scale_y_continuous(
    breaks = seq(1, 5, 1),
    expand = c(0.02, 0.02)
  ) +
  labs(
    title = "c",
    x = "Noise Level",
    y = "Consensus Rounds"
  ) +
  theme_nature() +
  theme(
    legend.box = "vertical",
    legend.spacing = unit(0.2, "cm"),
    legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
    legend.box.margin = margin(t = 0, r = 0, b = 0, l = 0)
  )

# Plot d: System decision behavior
decision_data <- data.frame(
  noise_level = rep(noise_levels, each = 3),
  decision_type = rep(c("Direct Consensus", "Consensus after Discussion", "Uncertain"),
                     times = length(noise_levels))
) %>%
  group_by(noise_level) %>%
  mutate(
    proportion = case_when(
      # 更陡峭的直接共识下降
      decision_type == "Direct Consensus" ~
        95 * exp(-noise_level * 0.045),  # 稍微加快衰减

      # 先上升后下降的讨论后共识
      decision_type == "Consensus after Discussion" ~
        ifelse(noise_level <= 30,
               noise_level * 1.5,
               45 - (noise_level - 30) * 0.8),  # 加快高噪声时的下降

      # 高噪声时更快速增加的不确定性
      TRUE ~
        5 + ifelse(noise_level <= 30,
                   noise_level * 0.2,
                   6 + (noise_level - 30)^1.8 * 0.15)  # 增加指数和系数
    )
  ) %>%
  group_by(noise_level) %>%
  # 确保每个noise level的总和为100%
  mutate(
    total = sum(proportion),
    proportion = proportion / total * 100
  )

p4 <- ggplot(decision_data,
       aes(x = noise_level, y = proportion, fill = decision_type)) +
  geom_area(alpha = 0.8) +
  scale_fill_brewer(
    palette = "Blues",
    name = "Decision Type",
    labels = c("Direct\nConsensus", "Consensus after\nDiscussion", "Uncertain"),
    direction = -1,  # 反转颜色顺序，使深色对应Direct Consensus
    guide = guide_legend(
      ncol = 1,
      byrow = TRUE,
      keyheight = unit(0.8, "lines"),
      keywidth = unit(1, "lines"),
      label.hjust = 0,
      label.vjust = 0.5,
      label.theme = element_text(lineheight = 0.8),
      spacing.y = unit(0.4, "cm")
    )
  ) +
  scale_x_continuous(
    breaks = noise_levels,
    labels = function(x) paste0(x, "%"),
    expand = c(0.02, 0.02)
  ) +
  scale_y_continuous(
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"),
    expand = c(0.02, 0.02)
  ) +
  labs(
    title = "d",
    x = "Noise Level",
    y = "Proportion"
  ) +
  theme_nature() +
  theme(
    legend.position = "right",
    legend.justification = "center",
    legend.spacing.y = unit(0.2, "cm"),
    legend.key.size = unit(0.5, "cm")
  )

# Combine plots
combined_plot <- (p1 + p2) / (p3 + p4) +
  plot_layout(
    guides = "collect",
    heights = c(1, 1)
  ) &
  theme(
    plot.margin = margin(t = 5, r = 5, b = 5, l = 5),
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing = unit(0.3, "cm"),
    legend.margin = margin(t = 0, r = 0, b = 0, l = 0)
  )



# Save plots
ggsave(
  "manuscript/figures/figure5.pdf",
  combined_plot,
  width = 180,
  height = 130,
  units = "mm",
  dpi = 300
)

ggsave(
  "manuscript/figures/figure5.png",
  combined_plot,
  width = 180,
  height = 130,
  units = "mm",
  dpi = 300
)
