# Set working directory to project root
project_root <- "."
library(tidyverse)

# Read the three files
l3_results <- read_csv(file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_3_results_v2.csv")) %>%
  select(reference_name, final_consensus) %>%
  rename(
    leiden_3 = reference_name,
    llm_ann_level_3 = final_consensus
  )

l4_results <- read_csv(file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_4_results_v2.csv")) %>%
  select(reference_name, final_consensus) %>%
  rename(
    leiden_4 = reference_name,
    llm_ann_level_4 = final_consensus
  )

original_ann <- read_csv(file.path(project_root, "data/popv_data/HLCA_combined_annotations.csv")) %>%
  select(
    leiden_4, leiden_3,
    original_ann_level_1, original_ann_level_2, original_ann_level_3,
    original_ann_level_4, original_ann_level_5, ann_level_1, ann_level_2,
    ann_level_3, ann_level_4, ann_level_5, reannotation_type
  ) %>%
  distinct(leiden_4, .keep_all = TRUE)  # 确保每个 leiden_4 只有一行

# Merge the tables，从 l4_results 开始
merged_results <- l4_results %>%
  left_join(original_ann, by = "leiden_4") %>%
  left_join(l3_results, by = "leiden_3")

# Save the merged results
write_csv(merged_results,
          file.path(project_root, "results/benchmark/popv_comparison/2_evaluation/HLCA_combined_annotations_with_consensus.csv"))

# Print preview
print("Preview of merged results:")
print(head(merged_results))
print("\nDimensions of merged results:")
print(dim(merged_results))
