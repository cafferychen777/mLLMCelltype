#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os

# Set paths
base_dir = "."
data_path = os.path.join(base_dir, "data/raw/HNOCA.h5ad")
llm_results_path = os.path.join(base_dir, "results/benchmark/reference_comparison/2_evaluation/HNOCA_L3_results.csv")
output_dir = os.path.join(base_dir, "results/benchmark/reference_comparison/2_evaluation")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read data
print("Reading data...")
adata = sc.read_h5ad(data_path)
llm_results = pd.read_csv(llm_results_path)

# Create standardized cell type names dictionary
standardized_names = {
    'Astrocyte': 'Astrocytes',
    'CP': 'Choroid plexus epithelial cells',
    'PSC': 'Choroid plexus epithelial cells',
    'Cerebellar NPC': 'Cerebellar Neural Progenitor Cells',
    'Cerebellar Neuron': 'Cerebellar Granule Neurons',
    'Dorsal Midbrain NPC': 'Dorsal Midbrain Neural Progenitor Cells',
    'Dorsal Midbrain Neuron': 'Dorsal Midbrain Immature Neuron',
    'Dorsal Telencephalic IP': 'Dorsal Telencephalon Intermediate Progenitor Cell',
    'Dorsal Telencephalic NPC': 'Dorsal telencephalon neural progenitor cells',
    'Dorsal Telencephalic Neuron': 'Dorsal Telencephalon Projection Neurons',
    'EC': 'Neural Progenitors',
    'Glioblast': 'Radial Glia',
    'Hypothalamic NPC': 'Hypothalamic Progenitors',
    'Hypothalamic Neuron': 'Hypothalamic Excitatory Neurons',
    'MC': 'Meningeal Fibroblasts',
    'Medulla NPC': 'Neural Progenitor Cells (Medulla Specific)',
    'Medulla Neuron': 'Medullary Neurons',
    'Microglia': 'Neural Macrophage/Microglia',
    'NC Derivatives': 'Schwann cells',
    'Neuroepithelium': 'Neural progenitor cells',
    'OPC': 'Oligodendrocyte Precursor Cells',
    'Pons NPC': 'Neural progenitor cells (Pons)',
    'Pons Neuron': 'Neurons (Pons)',
    'Thalamic NPC': 'Interneuron progenitors (Thalamus)',
    'Thalamic Neuron': 'Excitatory neurons of the Thalamus',
    'Ventral Midbrain NPC': 'Ventral Midbrain Progenitors',
    'Ventral Midbrain Neuron': 'Midbrain dopaminergic neurons',
    'Ventral Telencephalic NPC': 'Ventral Telencephalon Neural Progenitor Cells',
    'Ventral Telencephalic Neuron': 'Ventral Telencephalon Neuronal Progenitors'
}

def standardize_cell_type(cell_type):
    """Standardize cell type names using the mapping dictionary"""
    return standardized_names.get(cell_type, cell_type)

# Create prediction mapping using reference_name as the key
prediction_dict = dict(zip(llm_results['reference_name'], llm_results['final_consensus']))
controversy_dict = dict(zip(llm_results['reference_name'], llm_results['is_controversial']))

# Map annotations to standardized names
adata.obs['Reference'] = adata.obs['annot_level_3_rev2'].map(standardize_cell_type)
adata.obs['LLMCellType'] = adata.obs['annot_level_3_rev2'].map(prediction_dict)

# Calculate accuracy
print("\nCalculating accuracy metrics...")
y_true = adata.obs['Reference']
y_pred = adata.obs['LLMCellType']

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
class_report = classification_report(y_true, y_pred, zero_division=0)
print(class_report)

# Save results to file
results_df = pd.DataFrame({
    'Reference': y_true,
    'LLMCellType': y_pred,
    'Match': y_true == y_pred
})

# Add controversy information
results_df['Is_Controversial'] = adata.obs['annot_level_3_rev2'].map(controversy_dict)

# Calculate per-class accuracy
class_accuracy = results_df.groupby('Reference')['Match'].agg(['count', 'sum', 'mean']).round(4)
class_accuracy.columns = ['Total_Cells', 'Correct_Predictions', 'Accuracy']
class_accuracy = class_accuracy.sort_values('Accuracy', ascending=False)

# Calculate accuracy for controversial vs non-controversial predictions
controversial_accuracy = results_df[results_df['Is_Controversial']]['Match'].mean()
non_controversial_accuracy = results_df[~results_df['Is_Controversial']]['Match'].mean()

print(f"\nControversial Cases Accuracy: {controversial_accuracy:.4f}")
print(f"Non-controversial Cases Accuracy: {non_controversial_accuracy:.4f}")

# Save detailed results
output_file = os.path.join(output_dir, "HNOCA_accuracy_results.csv")
class_accuracy.to_csv(output_file)
print(f"\nDetailed results saved to: {output_file}")

# Print per-class accuracy
print("\nAccuracy by Cell Type:")
print(class_accuracy)
