#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import numpy as np
from collections import defaultdict

# Load data
print("Loading data...")
adata = sc.read_h5ad('data/processed/LCA_with_umap.h5ad')
results_df = pd.read_csv('results/benchmark/reference_comparison/2_evaluation/LCA_results.csv')

# Define standardization mappings (same as in visualization script)
cell_type_standardization = {
    # NK cells standardization
    'natural killer cell': 'NK cells',
    'mature NK T cell': 'NK cells',
    'NK cells': 'NK cells',
    'Natural killer cells': 'NK cells',

    # Macrophages standardization
    'alveolar macrophage': 'Macrophages',
    'macrophage': 'Macrophages',
    'Macrophage': 'Macrophages',
    'Macrophages': 'Macrophages',

    # Monocytes standardization
    'classical monocyte': 'Classical monocytes',
    'intermediate monocyte': 'Intermediate monocytes',
    'non-classical monocyte': 'Non-classical monocytes',
    'monocyte': 'Monocytes',
    'Classical monocytes': 'Classical monocytes',
    'Monocytes': 'Monocytes',

    # Fibroblasts standardization
    'fibroblast': 'Fibroblasts',
    'pulmonary interstitial fibroblast': 'Fibroblasts',
    'alveolar adventitial fibroblast': 'Fibroblasts',
    'Fibroblasts': 'Fibroblasts'
}

# Create mapping from original to consensus
cell_mapping = dict(zip(results_df['reference_name'], results_df['final_consensus']))

# Standardize cell types
def standardize_name(name):
    return cell_type_standardization.get(name, name)

# Create standardized mappings
standardized_original = defaultdict(set)
standardized_consensus = defaultdict(set)

# Process each cell
for idx in adata.obs.index:
    orig_type = adata.obs['cell_type'][idx]
    consensus_type = cell_mapping.get(orig_type, "Not found")

    # Standardize both names
    std_orig = standardize_name(orig_type)
    std_cons = standardize_name(consensus_type)

    # Map standardized cell barcode to both annotations
    standardized_original[idx] = std_orig
    standardized_consensus[idx] = std_cons

# Find inconsistencies
inconsistencies = defaultdict(lambda: {'count': 0, 'examples': []})
for idx in adata.obs.index:
    orig = standardized_original[idx]
    cons = standardized_consensus[idx]
    if orig != cons and cons != "Not found":
        key = f"{orig} -> {cons}"
        inconsistencies[key]['count'] += 1
        if len(inconsistencies[key]['examples']) < 5:  # Store up to 5 examples
            inconsistencies[key]['examples'].append(idx)

# Print analysis
print("\n=== Analysis of Inconsistencies After Standardization ===")
print("-" * 80)

# Sort inconsistencies by count
sorted_inconsistencies = sorted(inconsistencies.items(),
                              key=lambda x: x[1]['count'],
                              reverse=True)

total_cells = len(adata.obs.index)
total_inconsistent = sum(inc['count'] for inc in inconsistencies.values())

print(f"\nTotal cells analyzed: {total_cells}")
print(f"Total cells with inconsistent annotations: {total_inconsistent} ({total_inconsistent/total_cells*100:.2f}%)")
print(f"Number of different types of inconsistencies: {len(inconsistencies)}")

print("\nDetailed inconsistencies (sorted by frequency):")
print("-" * 80)
for mapping, data in sorted_inconsistencies:
    count = data['count']
    percentage = (count / total_cells) * 100
    examples = data['examples']

    print(f"\nMapping: {mapping}")
    print(f"Count: {count} cells ({percentage:.2f}% of total)")
    print("Example cell barcodes:")
    for ex in examples:
        print(f"  - {ex}")

# Save to CSV
output_df = pd.DataFrame([
    {
        'Original_Annotation': mapping.split(' -> ')[0],
        'Consensus_Annotation': mapping.split(' -> ')[1],
        'Cell_Count': data['count'],
        'Percentage': (data['count'] / total_cells) * 100,
        'Example_Barcodes': '|'.join(data['examples'])
    }
    for mapping, data in sorted_inconsistencies
])

output_file = 'results/benchmark/reference_comparison/3_visualization/annotation_inconsistencies.csv'
output_df.to_csv(output_file, index=False)
print(f"\nDetailed results saved to: {output_file}")

# Additional analysis of major discrepancies
print("\n=== Major Discrepancies Analysis ===")
print("-" * 80)
major_discrepancies = [inc for inc in sorted_inconsistencies if inc[1]['count'] > total_cells * 0.01]  # >1% of total cells
if major_discrepancies:
    print("\nMajor discrepancies (affecting >1% of cells):")
    for mapping, data in major_discrepancies:
        percentage = (data['count'] / total_cells) * 100
        print(f"\n{mapping}")
        print(f"Affects {data['count']} cells ({percentage:.2f}% of total)")
else:
    print("\nNo major discrepancies found (>1% of total cells)")
