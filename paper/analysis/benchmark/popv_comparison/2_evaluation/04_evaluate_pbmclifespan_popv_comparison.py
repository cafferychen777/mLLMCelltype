#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = 'results/figures/popv_comparison'
os.makedirs(output_dir, exist_ok=True)

# Define cell type mapping dictionaries
reference_to_standard = {
    # NK cells
    'NK_CD56_Dim': 'NK cells',
    'NK_CD56_Bright': 'NK cells',
    'NK_Proliferating': 'Proliferating NK cells',

    # Monocytes
    'Monocytes_CD14': 'Classical monocytes',
    'Monocytes_CD16': 'Non-classical monocytes',

    # CD4+ T cells
    'CD4_Naive_CCR7': 'Naive CD4+ T cells',
    'CD4_TCM_AQP3': 'CD4+ central memory T cells',
    'CD4_TEM_ANXA1': 'CD4+ effector memory T cells',
    'CD4_TEM_GNLY': 'CD4+ effector memory T cells',
    'CD4_Treg_FOXP3': 'Regulatory T cells',

    # CD8+ T cells
    'CD8_TEM_GNLY': 'CD8+ effector memory T cells',
    'CD8_Naive_LEF1': 'Naive CD8+ T cells',
    'CD8_TEM_CMC1': 'CD8+ effector memory T cells',
    'CD8_MAIT_SLC4A10': 'MAIT cells',
    'CD8_TCM_HAVCR2': 'CD8+ central memory T cells',
    'CD8_TEM_ZNF683': 'CD8+ effector memory T cells',

    # Other T cells
    'gdT': 'Gamma delta T cells',

    # B cells
    'B_BCR_GNLY': 'B cells',
    'B_Memory': 'Memory B cells',
    'B_Naive': 'Naive B cells',
    'B_Atypical_Memory': 'Atypical memory B cells',

    # Dendritic cells
    'mDC': 'Myeloid dendritic cells',
    'pDC': 'Plasmacytoid dendritic cells',

    # Other
    'Mega': 'Megakaryocytes',
    'Plasma cell': 'Plasma cells'
}

popv_to_standard = {
    # General cell types
    'granulocyte': 'Granulocytes',
    'T cell': 'T cells',
    'B cell': 'B cells',
    'monocyte': 'Monocytes',
    'hematopoietic cell': 'Hematopoietic cells',
    'leukocyte': 'Leukocytes',

    # T cells
    'CD8-positive, alpha-beta T cell': 'CD8+ T cells',
    'CD4-positive, alpha-beta T cell': 'CD4+ T cells',
    'CD4-positive, alpha-beta thymocyte': 'CD4+ T cells',
    'gamma-delta T cell': 'Gamma delta T cells',
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'Naive CD8+ T cells',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'Naive CD4+ T cells',
    'activated CD4-positive, alpha-beta T cell': 'CD4+ effector T cells',
    'activated CD8-positive, alpha-beta T cell': 'CD8+ effector T cells',
    'regulatory T cell': 'Regulatory T cells',
    'CD8-positive, alpha-beta thymocyte': 'CD8+ T cells',
    'mature NK T cell': 'NKT cells',
    'thymocyte': 'Thymocytes',

    # Monocytes and macrophages
    'classical monocyte': 'Classical monocytes',
    'non-classical monocyte': 'Non-classical monocytes',
    'intermediate monocyte': 'Intermediate monocytes',
    'macrophage': 'Macrophages',
    'colon macrophage': 'Macrophages',
    'tissue-resident macrophage': 'Macrophages',
    'microglial cell': 'Macrophages',

    # NK and innate cells
    'natural killer cell': 'NK cells',
    'innate lymphoid cell': 'NK cells',

    # Dendritic cells
    'myeloid dendritic cell': 'Myeloid dendritic cells',
    'plasmacytoid dendritic cell': 'Plasmacytoid dendritic cells',
    'Langerhans cell': 'Dendritic cells',

    # Precursor and stem cells
    'hematopoietic precursor cell': 'Hematopoietic cells',
    'hematopoietic stem cell': 'Hematopoietic cells',
    'common myeloid progenitor': 'Hematopoietic cells',
    'erythroid progenitor cell': 'Erythroid cells',

    # Other cell types
    'myeloid cell': 'Myeloid cells',
    'myeloid leukocyte': 'Myeloid cells',
    'neutrophil': 'Neutrophils',
    'mast cell': 'Mast cells',
    'platelet': 'Platelets',
    'plasma cell': 'Plasma cells',
    'basophil': 'Basophils',
    'erythrocyte': 'Erythrocytes',
    'erythroid lineage cell': 'Erythroid cells',
    'mononuclear phagocyte': 'Monocytes'
}

# Load the data
print("Loading the data...")
adata = sc.read_h5ad('results/benchmark/popv_comparison/2_evaluation/tcell_lifespan_popv_fast_with_umap.h5ad')
llm_results = pd.read_csv('results/benchmark/popv_comparison/2_evaluation/pbmc_lifespan_results.csv')

print("Processing the data...")

# Convert categorical columns to string first
for col in ['secondary_type', 'popv_prediction']:
    if pd.api.types.is_categorical_dtype(adata.obs[col]):
        adata.obs[col] = adata.obs[col].astype(str)

# Standardize reference cell types
adata.obs['secondary_type_standard'] = adata.obs['secondary_type'].map(reference_to_standard)

# For LLM predictions, use the final_consensus column from llm_results
# Create a mapping from reference_name to final_consensus
llm_predictions = dict(zip(llm_results['reference_name'], llm_results['final_consensus']))

# Map the predictions using secondary_type
adata.obs['llm_prediction_standard'] = adata.obs['secondary_type'].map(llm_predictions)

# Standardize PopV predictions
adata.obs['popv_prediction_standard'] = adata.obs['popv_prediction'].map(popv_to_standard)

# Detailed debugging information
print("\n=== DETAILED DEBUG INFORMATION ===")

# 1. Check data types
print("\n1. Data Types:")
print(f"secondary_type_standard dtype: {adata.obs['secondary_type_standard'].dtype}")
print(f"llm_prediction_standard dtype: {adata.obs['llm_prediction_standard'].dtype}")
print(f"popv_prediction_standard dtype: {adata.obs['popv_prediction_standard'].dtype}")

# 2. Check for NaN values and their sources
print("\n2. NaN Analysis:")

# Check original values that resulted in NaN after mapping
print("\nUnique values in secondary_type that resulted in NaN after mapping:")
mask_secondary = adata.obs['secondary_type_standard'].isna()
print(adata.obs.loc[mask_secondary, 'secondary_type'].value_counts().head(10))

print("\nUnique values in popv_prediction that resulted in NaN after mapping:")
mask_popv = adata.obs['popv_prediction_standard'].isna()
print(adata.obs.loc[mask_popv, 'popv_prediction'].value_counts().head(10))

# Summary of NaN counts
print("\nNaN Summary:")
print(f"Total cells: {len(adata.obs)}")
print(f"NaN in secondary_type_standard: {mask_secondary.sum()} ({mask_secondary.sum()/len(adata.obs)*100:.2f}%)")
print(f"NaN in llm_prediction_standard: {adata.obs['llm_prediction_standard'].isna().sum()} ({adata.obs['llm_prediction_standard'].isna().sum()/len(adata.obs)*100:.2f}%)")
print(f"NaN in popv_prediction_standard: {mask_popv.sum()} ({mask_popv.sum()/len(adata.obs)*100:.2f}%)")

# Print unique values in original columns for reference
print("\nAll unique values in original secondary_type:")
print(adata.obs['secondary_type'].unique())

print("\nAll unique values in original popv_prediction:")
print(adata.obs['popv_prediction'].unique())

# 3. Value distributions
print("\n3. Unique Values Distribution:")
print("\nTop 10 secondary_type_standard values:")
print(adata.obs['secondary_type_standard'].value_counts().head(10))
print("\nTop 10 llm_prediction_standard values:")
print(adata.obs['llm_prediction_standard'].value_counts().head(10))
print("\nTop 10 popv_prediction_standard values:")
print(adata.obs['popv_prediction_standard'].value_counts().head(10))

# 4. Check for whitespace issues
print("\n4. Sample Values with Lengths (first 5 of each):")
for col in ['secondary_type_standard', 'llm_prediction_standard', 'popv_prediction_standard']:
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
    'Reference': adata.obs['secondary_type_standard'],
    'LLM': adata.obs['llm_prediction_standard'],
    'PopV': adata.obs['popv_prediction_standard']
})

# Get unique mismatches
unique_pairs = set()
for idx, row in comparison_df[comparison_df['Reference'] != comparison_df['LLM']].iterrows():
    unique_pairs.add((row['Reference'], row['LLM']))

# Analyze similarity
print("\n=== Potential Synonym Analysis ===\n")

# Analyze LLM predictions
print("=== LLM Prediction Analysis ===\n")
llm_unique_pairs = set()
for idx, row in comparison_df[comparison_df['Reference'] != comparison_df['LLM']].iterrows():
    llm_unique_pairs.add((row['Reference'], row['LLM']))

for ref, pred in sorted(llm_unique_pairs):
    if ref and pred and isinstance(ref, str) and isinstance(pred, str):
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
for idx, row in comparison_df[comparison_df['Reference'] != comparison_df['PopV']].iterrows():
    popv_unique_pairs.add((row['Reference'], row['PopV']))

for ref, pred in sorted(popv_unique_pairs):
    if ref and pred and isinstance(ref, str) and isinstance(pred, str):
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
    # Full match (1.0)
    if is_equivalent_name(reference, prediction):
        return 1.0, 'Full match (exact or equivalent names)'

    # Partial match (0.5) based on hierarchy
    for parent, children in cell_type_hierarchy.items():
        # If prediction is parent and reference is child
        if prediction == parent and reference in children:
            return 0.5, f'Partial match (specific to general: {reference} -> {prediction})'
        # If reference is parent and prediction is child
        if reference == parent and prediction in children:
            return 0.5, f'Partial match (general to specific: {reference} -> {prediction})'
        # If both are children of same parent
        if reference in children and prediction in children:
            return 0.5, f'Partial match (sibling relationship under {parent})'

    # No match (0.0)
    return 0.0, 'No match'

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

# Calculate exact matches
llm_total = len(comparison_df)
llm_correct = (comparison_df['Reference'] == comparison_df['LLM']).sum()
llm_exact_accuracy = (llm_correct / llm_total) * 100

popv_total = len(comparison_df)
popv_correct = (comparison_df['Reference'] == comparison_df['PopV']).sum()
popv_exact_accuracy = (popv_correct / popv_total) * 100

# Calculate weighted accuracy using match scores
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
