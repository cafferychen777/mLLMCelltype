#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = 'results/figures/popv_comparison'
os.makedirs(output_dir, exist_ok=True)

# Define cell type mapping dictionaries
reference_to_standard = {
    # T cells and thymocytes
    'CD8-positive, alpha-beta T cell': 'CD8+ T cells',
    'CD4-positive, alpha-beta T cell': 'CD4+ T cells',
    'double-positive, alpha-beta thymocyte': 'Double positive thymocyte',
    'CD8-alpha-alpha-positive, alpha-beta intraepithelial T cell': 'CD8+ T cells',
    'regulatory T cell': 'Regulatory T cells',
    'Activated T cell': 'Activated T cells',
    'Activated T cells': 'Activated T cells',
    'T cell': 'T cells',
    'T cells (developing)': 'Early T cells',
    'lymphocyte': 'Lymphocytes',
    'CD8-positive, alpha-beta memory T cell': 'CD8+ memory T cells',
    'early T lineage precursor': 'Early T cells',
    'Early T cells/Thymocytes': 'Double negative thymocyte',
    'double negative thymocyte': 'Double negative thymocyte',
    'CD4-positive, alpha-beta memory T cell': 'CD4+ memory T cells',
    'alpha-beta T cell': 'Alpha-beta T cells',
    'gamma-delta T cell': 'Gamma-delta T cells',
    'Gamma delta T cells': 'Gamma-delta T cells',
    'Cortical Thymocytes': 'Double positive thymocyte',
    'CD8+ Naive T cells': 'CD8+ T cells',
    'Cytotoxic T cells': 'CD8+ T cells',
    'Thymocytes': 'Early T cells',
    'Regulatory T cell': 'Regulatory T cells',
    'CD8+ thymocytes': 'CD8+ T cells',
    'CD4+ thymocytes': 'CD4+ T cells',

    # Epithelial cells
    'cortical thymic epithelial cell': 'Cortical thymic epithelial cells',
    'medullary thymic epithelial cell': 'Medullary thymic epithelial cells',
    'epithelial cell of thymus': 'Thymic epithelial cells',
    'Thymic epithelial cell': 'Thymic epithelial cells',
    'Epithelial progenitor cells': 'Thymic epithelial cells',
    'Medullary thymic epithelial cell': 'Medullary thymic epithelial cells',
    'Cortical thymic epithelial cell': 'Cortical thymic epithelial cells',

    # B cells
    'memory B cell': 'Memory B cells',
    'naive B cell': 'Naive B cells',
    'precursor B cell': 'Precursor B cells',

    # Myeloid cells
    'dendritic cell': 'Dendritic cells',
    'plasmacytoid dendritic cell': 'Plasmacytoid dendritic cells',
    'macrophage': 'Macrophages',
    'monocyte': 'Monocytes',

    # Other immune cells
    'natural killer cell': 'Natural killer cells',
    'Natural Killer (NK) cells': 'Natural killer cells',
    'NK/NKT cells': 'Natural killer cells',
    'group 3 innate lymphoid cell': 'ILC3',

    # Other cells
    'fibroblast': 'Fibroblasts',
    'mast cell': 'Mast cells',
    'endothelial cell': 'Endothelial cells',
    'megakaryocyte': 'Megakaryocytes',
    'plasma cell': 'Plasma cells',
    'progenitor cell': 'Progenitor cells',
    'erythrocyte': 'Erythrocytes',
    'Erythroid precursors': 'Erythrocytes',
    'vascular associated smooth muscle cell': 'Vascular smooth muscle cells',
    'Smooth muscle cells': 'Vascular smooth muscle cells',
    'Monocytes/Macrophages': 'Macrophages',
    'Cytotoxic T cells': 'CD8+ T cells',
    'B cells (Immature/Pre-B cells)': 'Precursor B cells'
}

popv_to_standard = {
    # T cells and thymocytes
    'T follicular helper cell': 'T follicular helper cells',
    'thymocyte': 'Thymocytes',
    'CD4-positive, alpha-beta T cell': 'CD4+ T cells',
    'CD4-positive, alpha-beta thymocyte': 'CD4+ thymocytes',
    'CD8-positive, alpha-beta thymocyte': 'CD8+ thymocytes',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'CD4+ T cells',
    'medullary thymic epithelial cell': 'Medullary thymic epithelial cells',
    'CD8-positive, alpha-beta T cell': 'CD8+ T cells',
    'fibroblast': 'Fibroblasts',
    'regulatory T cell': 'Regulatory T cells',
    'T cell': 'T cells',
    'double-positive thymocyte': 'Double positive thymocyte',
    'double-positive, alpha-beta thymocyte': 'Double positive thymocyte',
    'double-negative thymocyte': 'Double negative thymocyte',
    'CD8-positive, alpha-beta memory T cell': 'CD8+ memory T cells',
    'plasma cell': 'Plasma cells',
    'mast cell': 'Mast cells',
    'cortical thymic epithelial cell': 'Cortical thymic epithelial cells',
    'CD4-positive, alpha-beta memory T cell': 'CD4+ memory T cells',
    'dendritic cell': 'Dendritic cells',
    'gamma-delta T cell': 'Gamma-delta T cells',
    'memory B cell': 'Memory B cells',
    'naive B cell': 'Naive B cells',
    'natural killer cell': 'Natural killer cells',
    'plasmacytoid dendritic cell': 'Plasmacytoid dendritic cells',
    'macrophage': 'Macrophages',
    'vascular associated smooth muscle cell': 'Vascular smooth muscle cells',
    'thymocyte': 'Thymocytes',
    'mature T cell': 'T cells',
    'immature T cell': 'Early T cells',
    'mature alpha-beta T cell': 'T cells',
    'mature thymocyte': 'T cells',
    'B cell': 'B cells',

    # Epithelial cells
    'cortical thymic epithelial cell': 'Cortical thymic epithelial cells',
    'medullary thymic epithelial cell': 'Medullary thymic epithelial cells',
    'epithelial cell': 'Thymic epithelial cells',

    # Specialized cell types
    'thymic fibroblast type 1': 'Fibroblasts',
    'thymic fibroblast type 2': 'Fibroblasts',
    'capillary endothelial cell': 'Endothelial cells',
    'lymphatic endothelial cell': 'Endothelial cells',
    'blood vessel endothelial cell': 'Endothelial cells',
    'high endothelial venule endothelial cell': 'Endothelial cells',

    # Other cells
    'fibroblast': 'Fibroblasts',
    'mast cell': 'Mast cells',
    'endothelial cell': 'Endothelial cells',
    'megakaryocyte': 'Megakaryocytes',
    'plasma cell': 'Plasma cells',
    'progenitor cell': 'Progenitor cells',
    'stromal cell': 'Fibroblasts',
    'connective tissue cell': 'Fibroblasts'
}

# Set up paths
base_dir = '.'
data_path = os.path.join(base_dir, "data/raw/Thymus.h5ad")
llm_results_path = os.path.join(base_dir, "results/benchmark/popv_comparison/2_evaluation/Thymus_results.csv")
popv_results_path = os.path.join(base_dir, "results/benchmark/popv_comparison/2_evaluation/Thymus_popv_fast_results.csv")

def standardize_cell_type(cell_type):
    """Standardize cell type names by removing extra spaces and standardizing separators"""
    if pd.isna(cell_type):
        return cell_type

    # Convert to string if needed
    cell_type = str(cell_type)

    # Convert to lowercase for processing
    cell_type = cell_type.lower()

    # Replace multiple spaces with single space
    cell_type = ' '.join(cell_type.split())

    # Handle various formats of alpha-beta
    cell_type = cell_type.replace('alpha beta', 'alpha-beta')
    cell_type = cell_type.replace('alphabeta', 'alpha-beta')
    cell_type = cell_type.replace('alpha,beta', 'alpha-beta')

    # Handle common variations
    cell_type = cell_type.replace(' - ', '-')
    cell_type = cell_type.replace('_', ' ')
    cell_type = cell_type.replace(',', '')

    # Standardize CD markers
    cell_type = cell_type.replace('cd4-positive', 'cd4+')
    cell_type = cell_type.replace('cd8-positive', 'cd8+')
    cell_type = cell_type.replace('cd4 positive', 'cd4+')
    cell_type = cell_type.replace('cd8 positive', 'cd8+')

    # Standardize common cell type terms
    replacements = [
        ('t-cell', 'T cell'),
        ('b-cell', 'B cell'),
        ('tcell', 'T cell'),
        ('bcell', 'B cell'),
        ('t cell', 'T cell'),
        ('b cell', 'B cell'),
        ('t cells', 'T cell'),
        ('b cells', 'B cell'),
        ('mast cell', 'mast cell'),  # Prevent incorrect capitalization
        ('nk cell', 'NK cell'),
        ('nkt cell', 'NKT cell')
    ]

    for old, new in replacements:
        cell_type = cell_type.replace(old, new)

    # Capitalize first letter of each word except for specific terms
    words = cell_type.split()
    skip_words = {'of', 'the', 'and', 'in', 'on', 'at', 'to'}

    for i, word in enumerate(words):
        if word not in skip_words or i == 0:
            if not any(marker in word for marker in ['cd4+', 'cd8+', 'NK', 'NKT']):
                words[i] = word.capitalize()

    return ' '.join(words)

# Load the data
print("Loading the data...")
adata = sc.read_h5ad(data_path)
llm_results = pd.read_csv(llm_results_path)

# Print original cell type distribution
print("\n=== Original Cell Type Distribution ===\n")
original_cell_types = adata.obs['cell_type'].value_counts()
print("Total unique cell types:", len(original_cell_types))
print("\nTop 20 most frequent cell types:")
print(original_cell_types.head(20))

# Check which cell types are missing from the mapping
print("\n=== Missing Mappings in reference_to_standard ===\n")
missing_types = set(adata.obs['cell_type'].unique()) - set(reference_to_standard.keys())
print(f"Found {len(missing_types)} unmapped cell types:")
for cell_type in sorted(missing_types):
    count = adata.obs[adata.obs['cell_type'] == cell_type].shape[0]
    print(f"{cell_type}: {count} cells")
popv_results = pd.read_csv(popv_results_path)

# Print some debug info about the data
print("\nDebug Info:")
print(f"adata.obs index type: {type(adata.obs.index)}")
print(f"adata.obs index dtype: {adata.obs.index.dtype}")
print(f"popv_results['barcodes'] type: {popv_results['barcodes'].dtype}")
print("Sample adata.obs index:", list(adata.obs.index[:5]))
print("Sample popv barcodes:", list(popv_results['barcodes'][:5]))

# Convert both to string and strip any whitespace
adata.obs.index = adata.obs.index.astype(str).str.strip()
popv_results['barcodes'] = popv_results['barcodes'].astype(str).str.strip()

# Create a mapping from barcode to prediction
barcode_to_prediction = dict(zip(popv_results['barcodes'], popv_results['popv_prediction']))

# Add predictions using the mapping
adata.obs['popv_prediction'] = adata.obs.index.map(barcode_to_prediction)

# Create a copy of the predictions
popv_predictions = adata.obs['popv_prediction'].copy()

# Print merge results
print(f"\nAfter adding predictions:")
print(f"NaN in popv_prediction: {adata.obs['popv_prediction'].isna().sum()} ({adata.obs['popv_prediction'].isna().mean():.2%})")

# Print sample of predictions to verify
print("\nSample of PopV predictions:")
print(adata.obs['popv_prediction'].value_counts().head())

# Standardize reference cell types
print("\nStandardizing cell types...")
adata.obs['cell_type_standard'] = adata.obs['cell_type'].map(reference_to_standard)

# Map LLM predictions
standardized_mapping = {}
for ref_name, final_cons in zip(llm_results['reference_name'], llm_results['final_consensus']):
    if pd.notna(ref_name) and pd.notna(final_cons):
        std_name = standardize_cell_type(ref_name)
        standardized_mapping[std_name] = final_cons

# Map predictions
adata.obs['llm_prediction_standard'] = adata.obs['cell_type'].apply(standardize_cell_type).map(standardized_mapping)

# Map PopV predictions
print("\nMapping PopV predictions...")
print("Sample PopV predictions before mapping:")
print(popv_predictions.value_counts().head())

adata.obs['popv_prediction_standard'] = popv_predictions.map(popv_to_standard)

print("\nSample PopV predictions after mapping:")
print(adata.obs['popv_prediction_standard'].value_counts().head())

# Print final NaN summary
# Analyze PopV prediction NaN values
print("\n=== PopV Prediction NaN Analysis ===\n")

# 1. Check original PopV predictions before standardization
print("1. Original PopV predictions distribution:")
popv_dist = adata.obs['popv_prediction'].value_counts()
print(popv_dist.head(20))
print(f"\nTotal unique PopV predictions: {len(adata.obs['popv_prediction'].unique())}")

# 2. Check which PopV predictions are not in mapping dictionary
print("\n2. PopV predictions not in mapping dictionary:")
unmapped_popv = set(adata.obs['popv_prediction'].unique()) - set(popv_to_standard.keys())
print(f"Found {len(unmapped_popv)} unmapped PopV predictions:")
for pred in sorted(unmapped_popv):
    count = adata.obs[adata.obs['popv_prediction'] == pred].shape[0]
    print(f"{pred}: {count} cells")

# 3. Analyze NaN values in popv_prediction_standard
print("\n3. Analysis of NaN values in popv_prediction_standard:")
nan_mask = adata.obs['popv_prediction_standard'].isna()
nan_predictions = adata.obs.loc[nan_mask, 'popv_prediction'].value_counts()
print("\nOriginal PopV predictions that resulted in NaN:")
for pred, count in nan_predictions.items():
    print(f"{pred}: {count} cells")
print(f"\nTotal cells with NaN: {nan_mask.sum()} ({nan_mask.mean():.2%})")

# 4. Print final NaN summary
print("\nFinal NaN Summary:")
print(f"NaN in cell_type_standard: {adata.obs['cell_type_standard'].isna().sum()} ({adata.obs['cell_type_standard'].isna().mean():.2%})")
print(f"NaN in llm_prediction_standard: {adata.obs['llm_prediction_standard'].isna().sum()} ({adata.obs['llm_prediction_standard'].isna().mean():.2%})")
print(f"NaN in popv_prediction_standard: {adata.obs['popv_prediction_standard'].isna().sum()} ({adata.obs['popv_prediction_standard'].isna().mean():.2%})")

print("Processing the data...")

# Convert categorical columns to string first
if pd.api.types.is_categorical_dtype(adata.obs['cell_type']):
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)

# Standardize reference cell types
adata.obs['cell_type_standard'] = adata.obs['cell_type'].map(reference_to_standard)

def standardize_cell_type(cell_type):
    """Standardize cell type names by removing extra spaces and standardizing separators"""
    if pd.isna(cell_type):
        return cell_type

    # Convert to string if needed
    cell_type = str(cell_type)

    # Convert to lowercase for processing
    cell_type = cell_type.lower()

    # Replace multiple spaces with single space
    cell_type = ' '.join(cell_type.split())

    # Handle various formats of alpha-beta
    cell_type = cell_type.replace('alpha beta', 'alpha-beta')
    cell_type = cell_type.replace('alphabeta', 'alpha-beta')
    cell_type = cell_type.replace('alpha,beta', 'alpha-beta')

    # Handle common variations
    cell_type = cell_type.replace(' - ', '-')
    cell_type = cell_type.replace('_', ' ')
    cell_type = cell_type.replace(',', '')

    # Standardize CD markers
    cell_type = cell_type.replace('cd4-positive', 'cd4+')
    cell_type = cell_type.replace('cd8-positive', 'cd8+')
    cell_type = cell_type.replace('cd4 positive', 'cd4+')
    cell_type = cell_type.replace('cd8 positive', 'cd8+')

    # Standardize common cell type terms
    replacements = [
        ('t-cell', 'T cell'),
        ('b-cell', 'B cell'),
        ('tcell', 'T cell'),
        ('bcell', 'B cell'),
        ('t cell', 'T cell'),
        ('b cell', 'B cell'),
        ('t cells', 'T cell'),
        ('b cells', 'B cell'),
        ('mast cell', 'mast cell'),  # Prevent incorrect capitalization
        ('nk cell', 'NK cell'),
        ('nkt cell', 'NKT cell')
    ]

    for old, new in replacements:
        cell_type = cell_type.replace(old, new)

    # Capitalize first letter of each word except for specific terms
    words = cell_type.split()
    skip_words = {'of', 'the', 'and', 'in', 'on', 'at', 'to'}

    for i, word in enumerate(words):
        if word not in skip_words or i == 0:
            if not any(marker in word for marker in ['cd4+', 'cd8+', 'NK', 'NKT']):
                words[i] = word.capitalize()

    return ' '.join(words)

# For LLM predictions, use the final_consensus column from llm_results
# Analyze NaN values
print("\n=== NaN Value Analysis ===\n")

# Original data NaN analysis
print("Original Data:")
print(f"Total cells: {len(adata.obs)}")
print(f"NaN in cell_type: {adata.obs['cell_type'].isna().sum()} ({adata.obs['cell_type'].isna().mean():.2%})")

# LLM results NaN analysis
print("\nLLM Results:")
print(f"Total rows in llm_results: {len(llm_results)}")
print(f"NaN in reference_name: {llm_results['reference_name'].isna().sum()} ({llm_results['reference_name'].isna().mean():.2%})")
print(f"NaN in final_consensus: {llm_results['final_consensus'].isna().sum()} ({llm_results['final_consensus'].isna().mean():.2%})")

# Standardization examples
print("\n=== Cell Type Standardization Examples ===\n")
for cell_type in adata.obs['cell_type'].unique()[:5]:
    std_type = standardize_cell_type(cell_type)
    print(f"Original: '{cell_type}'")
    print(f"Standardized: '{std_type}'\n")

# Create a mapping from standardized reference_name to final_consensus
standardized_mapping = {}
for ref_name, final_cons in zip(llm_results['reference_name'], llm_results['final_consensus']):
    if pd.notna(ref_name) and pd.notna(final_cons):  # Only add non-NaN values
        std_name = standardize_cell_type(ref_name)
        standardized_mapping[std_name] = final_cons

# Print mapping examples
print("=== Mapping Examples ===\n")
print("First 5 entries in standardized_mapping:")
for i, (key, value) in enumerate(list(standardized_mapping.items())[:5]):
    print(f"{i+1}. '{key}' -> '{value}'")

# Map the predictions using standardized cell_type
adata.obs['llm_prediction_standard'] = adata.obs['cell_type'].apply(standardize_cell_type).map(standardized_mapping)

# Standardize PopV predictions
adata.obs['popv_prediction_standard'] = popv_predictions.map(popv_to_standard)

# Detailed debugging information
print("\n=== DETAILED DEBUG INFORMATION ===")

# 1. Check data types
print("\n1. Data Types:")
print(f"cell_type_standard dtype: {adata.obs['cell_type_standard'].dtype}")
print(f"llm_prediction_standard dtype: {adata.obs['llm_prediction_standard'].dtype}")
print(f"popv_prediction_standard dtype: {adata.obs['popv_prediction_standard'].dtype}")

# 2. Check for NaN values and their sources
print("\n2. NaN Analysis:")

# Check original values that resulted in NaN after mapping
print("\nUnique values in cell_type that resulted in NaN after mapping:")
mask_cell_type = adata.obs['cell_type_standard'].isna()
print(adata.obs.loc[mask_cell_type, 'cell_type'].value_counts().head(10))

print("\nUnique values in popv_prediction that resulted in NaN after mapping:")
mask_popv = adata.obs['popv_prediction_standard'].isna()
popv_nan_values = adata.obs.loc[mask_popv, 'popv_prediction'].value_counts()
print("\nTop 10 unmapped PopV predictions:")
print(popv_nan_values.head(10))

print("\nTotal unique unmapped PopV cell types:", len(popv_nan_values))
print("\nTotal cells with unmapped PopV predictions:", mask_popv.sum())

# 计算每个未映射的PopV预测占总NaN的百分比
print("\nPercentage of total NaN values for each unmapped PopV prediction:")
for cell_type, count in popv_nan_values.head(10).items():
    percentage = (count / mask_popv.sum()) * 100
    print(f"{cell_type}: {count} cells ({percentage:.2f}%)")

# 检查这些未映射的细胞类型是否在popv_to_standard字典中
print("\nChecking if these cell types are in popv_to_standard dictionary:")
for cell_type in popv_nan_values.head(10).index:
    if cell_type in popv_to_standard:
        print(f"{cell_type}: Found in dictionary, maps to '{popv_to_standard[cell_type]}'")
    else:
        print(f"{cell_type}: Not found in dictionary")

# Summary of NaN counts
print("\nNaN Summary:")
print(f"Total cells: {len(adata.obs)}")
print(f"NaN in cell_type_standard: {mask_cell_type.sum()} ({mask_cell_type.sum()/len(adata.obs)*100:.2f}%)")
print(f"NaN in llm_prediction_standard: {adata.obs['llm_prediction_standard'].isna().sum()} ({adata.obs['llm_prediction_standard'].isna().sum()/len(adata.obs)*100:.2f}%)")
print(f"NaN in popv_prediction_standard: {mask_popv.sum()} ({mask_popv.sum()/len(adata.obs)*100:.2f}%)")

# Print unique values in original columns for reference
print("\nAll unique values in original cell_type:")
print(adata.obs['cell_type'].unique())

print("\nAll unique values in original popv_prediction:")
print(adata.obs['popv_prediction'].unique())

# 3. Value distributions
print("\n3. Unique Values Distribution:")
print("\nTop 10 cell_type_standard values:")
print(adata.obs['cell_type_standard'].value_counts().head(10))
print("\nTop 10 llm_prediction_standard values:")
print(adata.obs['llm_prediction_standard'].value_counts().head(10))
print("\nTop 10 popv_prediction_standard values:")
print(adata.obs['popv_prediction_standard'].value_counts().head(10))

# 4. Check for whitespace issues
print("\n4. Sample Values with Lengths (first 5 of each):")
for col in ['cell_type_standard', 'llm_prediction_standard', 'popv_prediction_standard']:
    print(f"\n{col}:")
    sample_values = adata.obs[col].head()
    for val in sample_values:
        if isinstance(val, str):
            print(f"Value: '{val}', Length: {len(val)}")
        else:
            print(f"Value: {val}, Type: {type(val)}")

from difflib import SequenceMatcher

def string_similarity(a, b):
    """Calculate string similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def normalize_cell_type(cell_type):
    """Normalize cell type names for comparison"""
    # Common replacements
    replacements = [
        ('nk', 'natural killer'),
        ('tem', 'effector memory'),
        ('tcm', 'central memory'),
        ('cd4+', 'cd4 positive'),
        ('cd8+', 'cd8 positive'),
        ('t cell', 't cells'),
        ('b cell', 'b cells'),
        ('memory b', 'b memory'),
        ('naive b', 'b naive')
    ]

    cell_type = cell_type.lower()
    for old, new in replacements:
        cell_type = cell_type.replace(old, new)
    return cell_type

# 5. Create detailed comparison DataFrame and analyze potential synonyms
comparison_df = pd.DataFrame({
    'Reference': adata.obs['cell_type_standard'],
    'LLM': adata.obs['llm_prediction_standard'],
    'PopV': adata.obs['popv_prediction_standard']
})

# Get unique mismatches (excluding NaN values)
llm_unique_pairs = set()
for idx, row in comparison_df.iterrows():
    if pd.notna(row['Reference']) and pd.notna(row['LLM']) and row['Reference'] != row['LLM']:
        llm_unique_pairs.add((str(row['Reference']), str(row['LLM'])))

# Analyze similarity
print("\n=== Potential Synonym Analysis ===\n")

# Analyze LLM predictions
print("=== LLM Prediction Analysis ===\n")

# Process and sort valid pairs
valid_pairs = [(ref, pred) for ref, pred in llm_unique_pairs
              if ref != 'nan' and pred != 'nan']

for ref, pred in sorted(valid_pairs):
    similarity = string_similarity(ref, pred)
    norm_ref = normalize_cell_type(ref)
    norm_pred = normalize_cell_type(pred)

    # Check if they're potential synonyms
    if similarity > 0.7 or norm_ref == norm_pred:
        print(f"Reference: '{ref}'")
        print(f"LLM Prediction: '{pred}'")
        print(f"Similarity Score: {similarity:.3f}")
        print(f"Normalized Reference: '{norm_ref}'")
        print(f"Normalized Prediction: '{norm_pred}'")
        print()

# Analyze PopV predictions
print("\n=== PopV Prediction Analysis ===\n")
popv_unique_pairs = set()
for idx, row in comparison_df.iterrows():
    if pd.notna(row['Reference']) and pd.notna(row['PopV']) and row['Reference'] != row['PopV']:
        popv_unique_pairs.add((str(row['Reference']), str(row['PopV'])))

# Process and sort valid pairs
valid_pairs = [(ref, pred) for ref, pred in popv_unique_pairs
              if ref != 'nan' and pred != 'nan']

for ref, pred in sorted(valid_pairs):
    similarity = string_similarity(ref, pred)
    norm_ref = normalize_cell_type(ref)
    norm_pred = normalize_cell_type(pred)

    # Check if they're potential synonyms
    if similarity > 0.7 or norm_ref == norm_pred:
        print(f"Reference: '{ref}'")
        print(f"PopV Prediction: '{pred}'")
        print(f"Similarity Score: {similarity:.3f}")
        print(f"Normalized Reference: '{norm_ref}'")
        print(f"Normalized Prediction: '{norm_pred}'")
        print()

# Define cell type hierarchy relationships for scoring
cell_type_hierarchy = {
    # Highest level
    'Hematopoietic cells': ['T cells', 'B cells', 'NK cells', 'Monocytes', 'Dendritic cells', 'Granulocytes', 'Megakaryocytes', 'Erythroid cells'],
    'Myeloid cells': ['Monocytes', 'Granulocytes', 'Dendritic cells', 'Megakaryocytes'],
    'Lymphoid cells': ['T cells', 'B cells', 'NK cells'],

    # T cells hierarchy
    'CD4+ T cells': ['CD4+ effector memory T cells', 'CD4+ central memory T cells', 'Naive CD4+ T cells', 'Regulatory T cells', 'Activated CD4+ T cells', 'CD4+ effector T cells'],
    'CD8+ T cells': ['CD8+ effector memory T cells', 'CD8+ central memory T cells', 'Naive CD8+ T cells', 'MAIT cells', 'Activated CD8+ T cells', 'CD8+ effector T cells'],
    'T cells': ['CD4+ T cells', 'CD8+ T cells', 'Gamma delta T cells', 'NKT cells', 'T helper cells', 'Thymocytes'],

    # B cells hierarchy
    'B cells': ['Memory B cells', 'Naive B cells', 'Atypical memory B cells', 'Plasma cells', 'Transitional B cells'],

    # NK cells hierarchy
    'NK cells': ['CD56 bright NK cells', 'CD56 dim NK cells', 'Proliferating NK cells'],

    # Monocytes and macrophages hierarchy
    'Monocytes': ['Classical monocytes', 'Non-classical monocytes', 'Intermediate monocytes'],
    'Macrophages': ['Tissue resident macrophages', 'Inflammatory macrophages'],

    # Dendritic cells hierarchy
    'Dendritic cells': ['Myeloid dendritic cells', 'Plasmacytoid dendritic cells', 'Conventional Dendritic Cells 1 (cDC1)', 'Conventional Dendritic Cells 2 (cDC2)', 'Langerhans cells'],
    'Myeloid dendritic cells': ['Conventional Dendritic Cells 1 (cDC1)', 'Conventional Dendritic Cells 2 (cDC2)'],

    # Granulocytes hierarchy
    'Granulocytes': ['Neutrophils', 'Eosinophils', 'Basophils', 'Mast cells']
}





# 6. Analyze mismatches
print("\n5. Detailed Mismatch Analysis:")
print("\nFirst 5 mismatches between Reference and LLM:")
mismatches = comparison_df[comparison_df['Reference'] != comparison_df['LLM']].head()
for idx, row in mismatches.iterrows():
    print(f"\nIndex: {idx}")
    print(f"Reference: '{row['Reference']}'")
    print(f"LLM: '{row['LLM']}'")
    print(f"PopV: '{row['PopV']}'")

# 7. Check exact matches
print("\n6. Exact Match Analysis:")
llm_exact_matches = (comparison_df['Reference'] == comparison_df['LLM']).sum()
print(f"Total exact matches with LLM: {llm_exact_matches}")
popv_exact_matches = (comparison_df['Reference'] == comparison_df['PopV']).sum()
print(f"Total exact matches with PopV: {popv_exact_matches}")



# Define equivalent names (Full match)
equivalent_names = {
    'NK cells': ['Natural Killer Cells'],
    'CD8+ effector memory T cells': ['Effector Memory CD8+ T cell (TEM)', 'CD8+ TEM'],
    'CD4+ effector memory T cells': ['CD4+ TEM'],
    'Classical monocytes': ['Classical Monocytes', 'CD14+ Monocytes'],
    'Non-classical monocytes': ['CD16+ Monocytes'],
    'Regulatory T cells': ['Regulatory T cells (Tregs)', 'Tregs', 'T regulatory cells'],
    'Gamma delta T cells': ['gdT', 'Gamma Delta T cells', 'γδ T cells'],
    'CD56 bright NK cells': ['NK_CD56_Bright', 'CD56bright NK cells'],
    'CD56 dim NK cells': ['NK_CD56_Dim', 'CD56dim NK cells'],
    'Proliferating NK cells': ['Proliferating cells'],
    'CD4+ central memory T cells': ['CD4+ TCM', 'Central Memory CD4+ T cells'],
    'CD8+ central memory T cells': ['CD8+ TCM', 'Central Memory CD8+ T cells'],
    'Conventional Dendritic Cells 1 (cDC1)': ['cDC1', 'Type 1 conventional dendritic cells'],
    'Conventional Dendritic Cells 2 (cDC2)': ['cDC2', 'Type 2 conventional dendritic cells']
}

# Create reverse mapping for equivalent names
reverse_equivalent_names = {}
for ref_type, equiv_list in equivalent_names.items():
    for equiv_type in equiv_list:
        reverse_equivalent_names[equiv_type] = ref_type

def is_equivalent_name(ref_type, pred_type):
    """Check if two cell type names are equivalent"""
    if ref_type == pred_type:
        return True
    # Check both directions
    if ref_type in equivalent_names and pred_type in equivalent_names[ref_type]:
        return True
    if pred_type in reverse_equivalent_names and ref_type == reverse_equivalent_names[pred_type]:
        return True
    return False

def calculate_match_score(reference, prediction):
    """Calculate match score between reference and prediction
    Returns:
    - score: 1.0 for full match, 0.5 for partial match, 0.0 for no match
    - match_type: description of the match type
    """
    # Handle NaN values
    if pd.isna(reference) or pd.isna(prediction):
        return 0.0, "NaN value"

    # Convert to string if needed and normalize
    reference = str(reference) if not isinstance(reference, str) else reference
    prediction = str(prediction) if not isinstance(prediction, str) else prediction
    ref_norm = reference.lower().replace('-', ' ').replace('(', '').replace(')', '')
    pred_norm = prediction.lower().replace('-', ' ').replace('(', '').replace(')', '')

    # Check for exact matches (including format variations)
    if ref_norm == pred_norm:
        return 1.0, 'Full match (exact)'

    # Check for singular/plural variations
    ref_singular = ref_norm.rstrip('s')
    pred_singular = pred_norm.rstrip('s')
    if ref_singular == pred_singular:
        return 1.0, 'Full match (singular/plural variation)'

    # Check for equivalent names with abbreviations
    abbrev_pairs = [
        ('natural killer cells', 'nk cells'),
        ('gamma delta t cells', 'γδ t cells'),
        ('alpha beta t cells', 'αβ t cells')
    ]
    for full, abbrev in abbrev_pairs:
        if (full in ref_norm and abbrev in pred_norm) or \
           (abbrev in ref_norm and full in pred_norm):
            return 1.0, 'Full match (abbreviation)'

    # Define cell state relationships
    state_relations = [
        # Memory/Naive/Activated T cell states
        ('cd4+ memory t cell', 'activated t cell', 'Cell state (memory/activated)'),
        ('cd8+ memory t cell', 'cytotoxic t cell', 'Cell state (memory/cytotoxic)'),
        # B cell states
        ('memory b cell', 'b cell', 'Cell state (memory B cell)'),
        ('naive b cell', 'b cell', 'Cell state (naive B cell)'),
        ('immature b cell', 'precursor b cell', 'Cell state (B cell development)'),
        # Myeloid cell relations
        ('monocyte', 'macrophage', 'Cell state (monocyte/macrophage)'),
    ]

    # Check for cell state relationships
    for state1, state2, desc in state_relations:
        if (state1 in ref_norm and state2 in pred_norm) or \
           (state2 in ref_norm and state1 in pred_norm):
            return 0.5, desc

    # Define developmental stage relationships
    developmental_stages = [
        # T cell development
        ('cd4+ thymocyte', 'cd4+ t cell', 'Development (thymocyte to T cell)'),
        ('cd8+ thymocyte', 'cd8+ t cell', 'Development (thymocyte to T cell)'),
        ('double negative thymocyte', 'early t cell', 'Development (DN thymocyte)'),
        ('double positive thymocyte', 'cortical thymocyte', 'Development (DP thymocyte)'),
        ('thymocyte', 'early t cell', 'Development (general to specific)'),
        ('developing t cell', 'early t cell', 'Development (T cell)'),
        ('double positive thymocyte', 'thymocyte', 'Development (DP to general)'),
        ('double negative thymocyte', 'thymocyte', 'Development (DN to general)'),
        ('double positive thymocyte', 'cd4+ thymocyte', 'Development (DP to CD4)'),
        ('double negative thymocyte', 'cd4+ thymocyte', 'Development (DN to CD4)'),
        # Epithelial cell development
        ('epithelial progenitor', 'thymic epithelial', 'Development (epithelial)'),
        ('cortical thymic epithelial', 'thymic epithelial', 'Development (epithelial)'),
        ('medullary thymic epithelial', 'thymic epithelial', 'Development (epithelial)'),
        # Other developmental relationships
        ('erythroid precursor', 'erythrocyte', 'Development (erythroid)'),
        ('b cell precursor', 'b cell', 'Development (B cell)')
    ]

    # Define T cell hierarchies
    t_cell_hierarchies = {
        't cell': ['t follicular helper cell', 'cd4+ t cell', 'cd8+ t cell',
                  'gamma delta t cell', 'alpha beta t cell', 'regulatory t cell',
                  'double positive thymocyte', 'double negative thymocyte'],
        'thymic t cell': ['double positive thymocyte', 'double negative thymocyte',
                         'cd4+ thymocyte', 'cd8+ thymocyte', 'early t cell'],
        'mature t cell': ['t follicular helper cell', 'cd4+ t cell', 'cd8+ t cell',
                         'gamma delta t cell', 'alpha beta t cell', 'regulatory t cell']
    }

    # Check for developmental stage relationships
    for stage1, stage2, desc in developmental_stages:
        if (stage1 in ref_norm and stage2 in pred_norm) or \
           (stage2 in ref_norm and stage1 in pred_norm):
            return 0.5, desc

    # Check T cell hierarchical relationships
    for parent, children in t_cell_hierarchies.items():
        # If both are in the same category
        if any(child in ref_norm for child in children) and \
           any(child in pred_norm for child in children):
            return 0.5, f'Hierarchy (shared {parent})'

    # Define cell type hierarchies
    cell_hierarchies = {
        't cell': ['cd4+ t cell', 'cd8+ t cell', 'gamma delta t cell', 'regulatory t cell',
                  'memory t cell', 'naive t cell', 'activated t cell', 'cytotoxic t cell'],
        'b cell': ['memory b cell', 'naive b cell', 'plasma cell', 'precursor b cell'],
        'myeloid cell': ['monocyte', 'macrophage', 'dendritic cell'],
        'lymphoid cell': ['t cell', 'b cell', 'nk cell', 'ilc']
    }

    # Check hierarchical relationships
    for parent, children in cell_hierarchies.items():
        if parent in ref_norm and any(child in pred_norm for child in children):
            return 0.5, f'Hierarchy (general to specific: {parent})'
        if parent in pred_norm and any(child in ref_norm for child in children):
            return 0.5, f'Hierarchy (specific to general: {parent})'

    return 0.0, "No match"

# Analyze LLM predictions
print("\n=== LLM Prediction Analysis ===\n")
llm_matches = []
for idx, row in comparison_df.iterrows():
    score, match_type = calculate_match_score(row['Reference'], row['LLM'])
    llm_matches.append({
        'Reference': row['Reference'],
        'Prediction': row['LLM'],
        'Score': score,
        'Match_Type': match_type
    })

llm_match_df = pd.DataFrame(llm_matches)

# Print LLM summary statistics
print("LLM Match Type Distribution:")
print(llm_match_df['Match_Type'].value_counts())
print("\nLLM Average Score:", llm_match_df['Score'].mean())

# Analyze PopV predictions
print("\n=== PopV Prediction Analysis ===\n")
popv_matches = []
for idx, row in comparison_df.iterrows():
    score, match_type = calculate_match_score(row['Reference'], row['PopV'])
    popv_matches.append({
        'Reference': row['Reference'],
        'Prediction': row['PopV'],
        'Score': score,
        'Match_Type': match_type
    })

popv_match_df = pd.DataFrame(popv_matches)

# Print PopV summary statistics
print("PopV Match Type Distribution:")
print(popv_match_df['Match_Type'].value_counts())
print("\nPopV Average Score:", popv_match_df['Score'].mean())

# Analyze mismatches in detail
print("\n=== Detailed Analysis of No Matches ===\n")
print("Top 20 most frequent LLM no-match pairs:")
llm_no_matches = llm_match_df[llm_match_df['Match_Type'] == 'No match']
llm_no_match_pairs = pd.DataFrame({
    'Reference': llm_no_matches['Reference'],
    'Prediction': llm_no_matches['Prediction']
}).value_counts().head(20)

for (ref, pred), count in llm_no_match_pairs.items():
    print(f"\nReference: '{ref}'")
    print(f"LLM Prediction: '{pred}'")
    print(f"Count: {count}")

print("\nTop 20 most frequent PopV no-match pairs:")
popv_no_matches = popv_match_df[popv_match_df['Match_Type'] == 'No match']
popv_no_match_pairs = pd.DataFrame({
    'Reference': popv_no_matches['Reference'],
    'Prediction': popv_no_matches['Prediction']
}).value_counts().head(20)

for (ref, pred), count in popv_no_match_pairs.items():
    print(f"\nReference: '{ref}'")
    print(f"PopV Prediction: '{pred}'")
    print(f"Count: {count}")

# Calculate final scores
llm_accuracy = llm_match_df['Score'].mean() * 100
popv_accuracy = popv_match_df['Score'].mean() * 100
improvement_percentage = llm_accuracy - popv_accuracy

# Calculate improvement percentage
improvement_percentage = llm_accuracy - popv_accuracy

# Calculate exact matches (excluding NaN values)
llm_valid = ~(comparison_df['Reference'].isna() | comparison_df['LLM'].isna())
llm_total = llm_valid.sum()
llm_correct = (comparison_df[llm_valid]['Reference'] == comparison_df[llm_valid]['LLM']).sum()
llm_exact_accuracy = (llm_correct / llm_total) * 100 if llm_total > 0 else 0

popv_valid = ~(comparison_df['Reference'].isna() | comparison_df['PopV'].isna())
popv_total = popv_valid.sum()
popv_correct = (comparison_df[popv_valid]['Reference'] == comparison_df[popv_valid]['PopV']).sum()
popv_exact_accuracy = (popv_correct / popv_total) * 100 if popv_total > 0 else 0

# Calculate weighted accuracy using match scores (NaN values already handled in calculate_match_score)
llm_accuracy = llm_match_df['Score'].mean() * 100
popv_accuracy = popv_match_df['Score'].mean() * 100
improvement_percentage = llm_accuracy - popv_accuracy

print(f"\nDetailed Accuracy Results:")
print(f"LLMCelltype: {llm_correct}/{llm_total} exact matches ({llm_exact_accuracy:.2f}%), weighted accuracy: {llm_accuracy:.2f}%")
print(f"PopV: {popv_correct}/{popv_total} exact matches ({popv_exact_accuracy:.2f}%), weighted accuracy: {popv_accuracy:.2f}%")
print(f"Improvement: {improvement_percentage:.2f}%")

# Save results to a text file
with open(os.path.join(output_dir, 'pbmc_lifespan_accuracy.txt'), 'w') as f:
    f.write(f"=== Detailed Accuracy Results ===\n")
    f.write(f"LLMCelltype:\n")
    f.write(f"  - Exact matches: {llm_correct}/{llm_total} ({llm_exact_accuracy:.2f}%)\n")
    f.write(f"  - Weighted accuracy: {llm_accuracy:.2f}%\n")
    f.write(f"\nPopV:\n")
    f.write(f"  - Exact matches: {popv_correct}/{popv_total} ({popv_exact_accuracy:.2f}%)\n")
    f.write(f"  - Weighted accuracy: {popv_accuracy:.2f}%\n")
    f.write(f"\nImprovement: {improvement_percentage:.2f}%\n")
