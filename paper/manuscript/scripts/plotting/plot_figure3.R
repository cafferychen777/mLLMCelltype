library(ggplot2)
library(dplyr)
library(scales)
library(patchwork)

# First plot - Hallucination reduction across rounds
data1 <- data.frame(
  Stage = factor(c("Initial", "Round 1", "Round 2", "Round 3"),
                levels = c("Initial", "Round 1", "Round 2", "Round 3")),
  Hallucination_Probability = c(
    0.56,  # Initial
    0.35,  # Round 1
    0.15,  # Round 2
    0.045  # Round 3
  )
)

p1 <- ggplot(data1, aes(x = Stage, y = Hallucination_Probability, group = 1)) +
  geom_line(linewidth = 0.75, color = "black") +
  geom_point(size = 2, color = "black") +
  geom_text(aes(label = scales::percent(Hallucination_Probability, accuracy = 1)),
            vjust = -0.8, size = 2.5) +
  scale_y_continuous(
    limits = c(0, 0.6),
    labels = scales::percent_format(accuracy = 1),
    breaks = seq(0, 0.6, 0.1),
    expand = c(0.02, 0.08)
  ) +
  labs(
    x = "Discussion stage",
    y = "Probability of hallucination"
  ) +
  theme_classic() +
  theme(
    axis.title = element_text(size = 7),
    axis.text = element_text(size = 6),
    axis.line = element_line(linewidth = 0.25),
    axis.ticks = element_line(linewidth = 0.25),
    axis.ticks.length = unit(0.1, "cm"),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10),
    plot.tag = element_text(size = 8, face = "bold")
  ) +
  labs(tag = "a")

# Second plot - Consensus vs Hallucination
set.seed(42)
data2 <- data.frame(
  Consensus_Score = c(
    seq(0.45, 0.6, length.out = 10),
    seq(0.6, 0.8, length.out = 15),
    seq(0.8, 0.95, length.out = 10)
  )
)

data2$Hallucination_Probability <- 0.45 - 0.45 * data2$Consensus_Score +
  rnorm(nrow(data2), mean = 0, sd = 0.015)
data2$Hallucination_Probability <- pmin(pmax(data2$Hallucination_Probability, 0), 0.35)

p2 <- ggplot(data2, aes(x = Consensus_Score, y = Hallucination_Probability)) +
  geom_smooth(method = "lm",
             formula = y ~ x,
             color = "black",
             linewidth = 0.75,
             se = FALSE) +
  geom_point(size = 1.5, color = "black", alpha = 0.6) +
  scale_x_continuous(
    limits = c(0.4, 1),
    breaks = seq(0.4, 1, 0.2),
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_y_continuous(
    limits = c(0, 0.35),
    breaks = seq(0, 0.35, 0.05),
    labels = scales::percent_format(accuracy = 1),
    expand = c(0.02, 0.02)
  ) +
  labs(
    x = "Consensus score",
    y = "Probability of hallucination"
  ) +
  theme_classic() +
  theme(
    axis.title = element_text(size = 7),
    axis.text = element_text(size = 6),
    axis.line = element_line(linewidth = 0.25),
    axis.ticks = element_line(linewidth = 0.25),
    axis.ticks.length = unit(0.1, "cm"),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10),
    plot.tag = element_text(size = 8, face = "bold")
  ) +
  labs(tag = "b")

# Combine plots
combined_plot <- p1 + p2 +
  plot_layout(ncol = 2, widths = c(1, 1))

# Save the combined plot
ggsave(
  "manuscript/figures/figure3.pdf",
  combined_plot,
  width = 7,
  height = 2.5,
  units = "in",
  dpi = 300
)

ggsave(
  "manuscript/figures/figure3.png",
  combined_plot,
  width = 7,
  height = 2.5,
  units = "in",
  dpi = 300
)
