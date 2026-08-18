#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# Read the original results
results_df = pd.read_csv('results/benchmark/reference_comparison/2_evaluation/LCA_results.csv')

# Extract only the necessary columns
final_consensus_df = results_df[['cluster_id', 'final_consensus']].copy()

# Clean the final_consensus column by removing the numeric prefix
final_consensus_df['final_consensus'] = final_consensus_df['final_consensus'].str.replace(r'^\d+:\s*', '', regex=True)

# Save to a new CSV file
output_file = 'results/benchmark/reference_comparison/2_evaluation/LCA_final_consensus.csv'
final_consensus_df.to_csv(output_file, index=False)
print(f"Final consensus saved to: {output_file}")
print("\nFirst few rows of the final consensus:")
print(final_consensus_df.head())
