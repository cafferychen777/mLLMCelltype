#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = 'results/figures/popv_comparison'
os.makedirs(output_dir, exist_ok=True)

# Set data paths
data_path = 'data/processed/LCA_with_umap.h5ad'
llm_results_path = 'results/benchmark/reference_comparison/2_evaluation/LCA_results.csv'
popv_results_path = 'results/benchmark/popv_comparison/2_evaluation/LCA_popv_fast_results.csv'

# Define cell type mapping dictionaries based on LCA standardization
reference_to_standard = {
    # Immune cells
    'natural killer cell': 'NK cells',
    'NK cells': 'NK cells',
    'mature NK T cell': 'NK cells',
    'B cell': 'B cells',
    'plasma cell': 'Plasma cells',
    'lymphocyte': 'Lymphocytes',
    'myeloid leukocyte': 'Myeloid cells',
    'alveolar macrophage': 'Macrophages',
    'macrophage': 'Macrophages',

    # AT1/AT2 cells
    'pulmonary alveolar type 1 cell': 'AT1 cells',
    'pulmonary alveolar type 2 cell': 'AT2 cells',
    'alveolar type i cell': 'AT1 cells',
    'alveolar type ii cell': 'AT2 cells',
    'type i pneumocyte': 'AT1 cells',
    'type ii pneumocyte': 'AT2 cells',
    'alveolar epithelial type i cell': 'AT1 cells',
    'alveolar epithelial type ii cell': 'AT2 cells',
    'alveolar type ii pneumocyte': 'AT2 cells',

    # T cells
    'effector memory CD4-positive, alpha-beta T cell': 'CD4+ memory T cells',
    'effector memory CD8-positive, alpha-beta T cell': 'CD8+ memory T cells',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'CD4+ naive T cells',
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'CD8+ naive T cells',

    # Monocytes
    'classical monocyte': 'Classical monocytes',
    'non-classical monocyte': 'Non-classical monocytes',
    'intermediate monocyte': 'Intermediate monocytes',
    'monocyte': 'Monocytes',

    # Dendritic cells
    'dendritic cell': 'Dendritic cells',
    'CD1c-positive myeloid dendritic cell': 'Myeloid dendritic cells',
    'myeloid dendritic cell, human': 'Myeloid dendritic cells',
    'plasmacytoid dendritic cell, human': 'Plasmacytoid dendritic cells',

    # Epithelial cells
    'epithelial cell': 'Epithelial cells',
    'basal cell': 'Basal epithelial cells',
    'respiratory basal cell': 'Basal epithelial cells',
    'club cell': 'Club cells',
    'ciliated cell': 'Ciliated epithelial cells',
    'lung ciliated cell': 'Ciliated epithelial cells',
    'lung goblet cell': 'Goblet cells',
    'lung neuroendocrine cell': 'Neuroendocrine cells',
    'mucus secreting cell': 'Secretory cells',
    'tracheobronchial serous cell': 'Secretory cells',
    'pulmonary ionocyte': 'Ionocytes',
    'pulmonary alveolar type 1 cell': 'AT1 cells',
    'pulmonary alveolar type 2 cell': 'AT2 cells',
    'mesothelial cell of pleura': 'Mesothelial cells',

    # Endothelial cells
    'endothelial cell': 'Endothelial cells',
    'blood vessel endothelial cell': 'Endothelial cells',
    'vascular endothelial cell': 'Endothelial cells',
    'endothelial cell of blood vessel': 'Endothelial cells',
    'capillary endothelial cell': 'Capillary endothelial cells',
    'lung capillary endothelial cell': 'Capillary endothelial cells',
    'pulmonary capillary endothelial cell': 'Capillary endothelial cells',
    'endothelial cell of artery': 'Arterial endothelial cells',
    'arterial endothelial cell': 'Arterial endothelial cells',
    'artery endothelial cell': 'Arterial endothelial cells',
    'vein endothelial cell': 'Venous endothelial cells',
    'venous endothelial cell': 'Venous endothelial cells',
    'endothelial cell of vein': 'Venous endothelial cells',
    'endothelial cell of lymphatic vessel': 'Lymphatic endothelial cells',
    'lymphatic endothelial cell': 'Lymphatic endothelial cells',
    'lymph vessel endothelial cell': 'Lymphatic endothelial cells',
    'lymphatic vessel endothelial cell': 'Lymphatic endothelial cells',

    # Stromal cells
    'fibroblast': 'Fibroblasts',
    'pulmonary interstitial fibroblast': 'Fibroblasts',
    'myofibroblast cell': 'Myofibroblasts',
    'pericyte': 'Pericytes',
    'bronchial smooth muscle cell': 'Smooth muscle cells',
    'vascular associated smooth muscle cell': 'Smooth muscle cells',

    # Other cell types
    'megakaryocyte': 'Megakaryocytes'
}

popv_to_standard = {
    # NK cells
    'natural killer cell': 'NK cells',
    'NK cell': 'NK cells',
    'mature NK T cell': 'NK cells',

    # T cells - CD4+ and CD8+
    'cd8-positive, alpha-beta t cell': 'CD8+ Alpha-beta T Cell',
    'cd4-positive, alpha-beta t cell': 'CD4+ Alpha-beta T Cell',
    'cd8+ alpha-beta t cell': 'CD8+ Alpha-beta T Cell',
    'cd4+ alpha-beta t cell': 'CD4+ Alpha-beta T Cell',
    'cd8+ t cell': 'CD8+ T cells',
    'cd4+ t cell': 'CD4+ T cells',
    'cd8-positive t cell': 'CD8+ T cells',
    'cd4-positive t cell': 'CD4+ T cells',
    'cd8 t cell': 'CD8+ T cells',
    'cd4 t cell': 'CD4+ T cells',

    # T cell states
    'naive cd8+ t cell': 'CD8+ naive T cells',
    'naive cd4+ t cell': 'CD4+ naive T cells',
    'memory cd8+ t cell': 'CD8+ memory T cells',
    'memory cd4+ t cell': 'CD4+ memory T cells',
    'effector cd8+ t cell': 'CD8+ effector T cells',
    'effector cd4+ t cell': 'CD4+ effector T cells',

    # T cells
    'CD8-positive, alpha-beta T cell': 'CD8+ T cells',
    'CD4-positive, alpha-beta T cell': 'CD4+ T cells',

    # Macrophages and Monocytes
    'alveolar macrophage': 'Macrophages',
    'macrophage': 'Macrophages',
    'classical monocyte': 'Classical monocytes',
    'intermediate monocyte': 'Intermediate monocytes',
    'non-classical monocyte': 'Non-classical monocytes',
    'monocyte': 'Monocytes',

    # Dendritic cells
    'myeloid dendritic cell': 'Dendritic cells',
    'plasmacytoid dendritic cell': 'Plasmacytoid dendritic cells',

    # Epithelial cells
    'club cell': 'Club cells',
    'lung ciliated cell': 'Ciliated epithelial cells',
    'basal cell': 'Basal epithelial cells',
    'pulmonary alveolar type 1 cell': 'AT1 cells',
    'pulmonary alveolar type 2 cell': 'AT2 cells',
    'pulmonary ionocyte': 'Ionocytes',
    'respiratory goblet cell': 'Goblet cells',
    'serous cell of epithelium of bronchus': 'Serous cells',

    # Endothelial cells
    'endothelial cell': 'Endothelial cells',
    'capillary endothelial cell': 'Capillary endothelial cells',
    'endothelial cell of artery': 'Arterial endothelial cells',
    'endothelial cell of lymphatic vessel': 'Lymphatic endothelial cells',
    'vein endothelial cell': 'Venous endothelial cells',

    # Stromal cells
    'fibroblast': 'Fibroblasts',
    'alveolar adventitial fibroblast': 'Fibroblasts',
    'adventitial cell': 'Fibroblasts',
    'pericyte': 'Pericytes',
    'smooth muscle cell': 'Smooth muscle cells',
    'bronchial smooth muscle cell': 'Smooth muscle cells',
    'vascular associated smooth muscle cell': 'Smooth muscle cells',

    # Other cell types
    'mast cell': 'Mast cells',
    'neutrophil': 'Neutrophils',
    'plasma cell': 'Plasma cells',
    'B cell': 'B cells',
    'basophil': 'Basophils',
    'mesothelial cell': 'Mesothelial cells'
}

# Load the data
print("Loading the data...")

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
        ('nkt cell', 'NKT cell'),
        # Add endothelial cell variations
        ('endothelial cells', 'Endothelial Cell'),
        ('endothelial cell', 'Endothelial Cell'),
        ('blood vessel endothelial cell', 'Blood Vessel Endothelial Cell'),
        ('vascular endothelial cell', 'Vascular Endothelial Cell'),
        ('capillary endothelial cell', 'Capillary Endothelial Cell'),
        ('arterial endothelial cell', 'Arterial Endothelial Cell'),
        ('lymphatic endothelial cell', 'Lymphatic Endothelial Cell')
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

    final_result = ' '.join(words)

    return final_result

# Load and process the data
print("Loading and processing data...")
adata = sc.read_h5ad(data_path)
llm_results = pd.read_csv(llm_results_path)
popv_results = pd.read_csv(popv_results_path)

# Set index for PopV results
popv_results.set_index('index', inplace=True)

# Print data shapes
print(f"\nData shapes:")
print(f"adata shape: {adata.shape}")
print(f"PopV results shape: {popv_results.shape}")
print(f"LLM results shape: {llm_results.shape}")

# Print sample indices
print("\nSample indices:")
print("adata.obs index:")
print(adata.obs.index[:5].tolist())
print("\npopv_results index:")
print(popv_results.index[:5].tolist())

# Check index overlap
overlapping_cells = len(set(adata.obs.index) & set(popv_results.index))
print(f"\nIndex overlap: {overlapping_cells} cells")

# Convert categorical columns to string
if pd.api.types.is_categorical_dtype(adata.obs['cell_type']):
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)

# Print original cell type distribution
print("\n=== Original Cell Type Distribution ===\n")
print("Total unique cell types:", len(adata.obs['cell_type'].unique()))
print("\nTop 20 most frequent cell types:")
print(adata.obs['cell_type'].value_counts().head(20))

# Process predictions
print("\nProcessing predictions...")

# 1. Standardize reference cell types
print("\n1. Standardizing reference cell types...")
adata.obs['cell_type_standard'] = adata.obs['cell_type'].map(reference_to_standard)

# 2. Add LLM predictions
print("2. Processing LLM predictions...")
# Create mapping with both full and truncated T cell names
llm_mapping = dict(zip(llm_results['reference_name'], llm_results['final_consensus']))
# Add full T cell names mapping
llm_mapping.update({
    'naive thymus-derived CD8-positive, alpha-beta T cell': llm_mapping.get('naive thymus-derived CD8-positive'),
    'naive thymus-derived CD4-positive, alpha-beta T cell': llm_mapping.get('naive thymus-derived CD4-positive'),
    'effector memory CD4-positive, alpha-beta T cell': llm_mapping.get('effector memory CD4-positive'),
    'effector memory CD8-positive, alpha-beta T cell': llm_mapping.get('effector memory CD8-positive')
})
adata.obs['llm_prediction'] = adata.obs['cell_type'].map(llm_mapping)
adata.obs['llm_prediction_standard'] = adata.obs['llm_prediction'].apply(lambda x: reference_to_standard.get(standardize_cell_type(x), x) if pd.notna(x) else x)

# 3. Add PopV predictions
print("3. Processing PopV predictions...")

# 检查 PopV 预测的唯一值
print("\n=== PopV 预测的详细分析 ===\n")

# 1. 获取所有唯一的预测结果
all_predictions = popv_results['popv_prediction'].unique()
print(f"总共有 {len(all_predictions)} 种不同的预测结果")

# 2. 检查每个预测结果的标准化情况
print("\n预测结果的标准化分析:")
standardization_results = {}
for pred in all_predictions:
    if pd.isna(pred):
        continue
    standardized = standardize_cell_type(pred)
    in_dict = standardized.lower() in {k.lower(): v for k, v in popv_to_standard.items()}
    standardization_results[pred] = {
        '原始预测': pred,
        '标准化后': standardized,
        '是否在字典中': in_dict,
        '出现频率': len(popv_results[popv_results['popv_prediction'] == pred])
    }

# 3. 分类统计
print("\n=== 预测结果统计 ===\n")
total_cells = len(popv_results)
mapped_cells = sum(info['出现频率'] for info in standardization_results.values() if info['是否在字典中'])
unmapped_cells = sum(info['出现频率'] for info in standardization_results.values() if not info['是否在字典中'])

print(f"总细胞数: {total_cells:,}")
print(f"可成功映射的细胞数: {mapped_cells:,} ({mapped_cells/total_cells*100:.2f}%)")
print(f"未能映射的细胞数: {unmapped_cells:,} ({unmapped_cells/total_cells*100:.2f}%)")

# 4. 显示未映射的细胞类型
print("\n=== 未被映射的细胞类型 ===\n")
unmapped_types = [(k, v) for k, v in standardization_results.items() if not v['是否在字典中']]
unmapped_types.sort(key=lambda x: x[1]['出现频率'], reverse=True)

print("按出现频率降序排列 (仅显示出现超过100次的类型):")
print("-" * 80)
print(f"{'原始预测':<40} {'标准化后':<40} {'出现次数':<10}")
print("-" * 80)
for cell_type, info in unmapped_types:
    if info['出现频率'] > 100:
        print(f"{info['原始预测']:<40} {info['标准化后']:<40} {info['出现频率']:<10,d}")

# 5. 建议添加到字典中的映射
print("\n=== 建议添加到 popv_to_standard 字典中的映射 ===\n")
print("以下是建议的映射关系，请根据生物学知识进行审查和修改：\n")
print("```python")
print("# 添加到 popv_to_standard 字典")
for cell_type, info in unmapped_types:
    if info['出现频率'] > 100:
        suggested_standard = info['标准化后']
        print(f"    '{suggested_standard.lower()}': '{suggested_standard}',  # 出现 {info['出现频率']:,} 次")
print("```")

# 检查每个预测方法的 NaN 值
for method in ['popv_svm_prediction', 'popv_xgboost_prediction', 'popv_onclass_prediction', 'popv_celltypist_prediction']:
    if method in popv_results.columns:
        print(f"\n{method} NaN 分析:")
        print(f"NaN 值数量: {popv_results[method].isna().sum()}")
        print(f"'nan' 字符串数量: {(popv_results[method] == 'nan').sum()}")

# 检查 majority vote 的结果
if 'popv_majority_vote_prediction' in popv_results.columns:
    print("\nMajority Vote 预测分析:")
    print(f"NaN 值数量: {popv_results['popv_majority_vote_prediction'].isna().sum()}")
    print(f"'nan' 字符串数量: {(popv_results['popv_majority_vote_prediction'] == 'nan').sum()}")

# 查看 PopV 预测的唯一值
print("\nPopV 预测中的唯一值:")
print(popv_results['popv_prediction'].unique())

# 添加 PopV 预测到 adata
adata.obs['popv_prediction'] = popv_results['popv_prediction']

# 检查合并后的数据
print("\n合并后的数据分析:")
print(f"合并后总行数: {len(adata.obs)}")
print(f"合并后 PopV 预测中的 NaN 值数量: {adata.obs['popv_prediction'].isna().sum()}")
print(f"合并后 PopV 预测中的 'nan' 字符串数量: {(adata.obs['popv_prediction'] == 'nan').sum()}")

# 标准化 PopV 预测
adata.obs['popv_prediction_standard'] = adata.obs['popv_prediction'].apply(lambda x: popv_to_standard.get(standardize_cell_type(x), x) if pd.notna(x) else x)

# Print NaN summary
print("\n=== NaN Summary ===\n")
for col in ['cell_type_standard', 'llm_prediction_standard', 'popv_prediction_standard']:
    nan_count = adata.obs[col].isna().sum()
    print(f"NaN in {col}: {nan_count} ({nan_count/len(adata)*100:.2f}%)")

# 详细分析 PopV 预测标签
print("\n=== PopV 预测标签详细分析 ===\n")

# 1. 获取所有 PopV 预测的唯一标签
popv_labels = adata.obs['popv_prediction'].dropna().unique()
print(f"PopV 总共预测了 {len(popv_labels)} 种不同的细胞类型")

# 2. 分析未映射的标签
unmapped_labels = set(popv_labels) - set(popv_to_standard.keys())
print(f"\n未被映射的标签数量: {len(unmapped_labels)}")

# 3. 统计每个未映射标签的出现次数
print("\n未映射标签及其出现次数:")
unmapped_counts = adata.obs[adata.obs['popv_prediction'].isin(unmapped_labels)]['popv_prediction'].value_counts()
for label, count in unmapped_counts.items():
    print(f"{label}: {count} cells")

# 4. 分析标准化后的结果
print("\n标准化后的标签分布:")
std_counts = adata.obs['popv_prediction_standard'].value_counts()
print("\n前20个最常见的标准化标签:")
for label, count in std_counts.head(20).items():
    print(f"{label}: {count} cells")

# 5. 分析映射字典的覆盖率
print(f"\n映射字典覆盖率分析:")
total_cells = len(adata)
mapped_cells = adata.obs['popv_prediction'].isin(popv_to_standard.keys()).sum()
print(f"总细胞数: {total_cells}")
print(f"成功映射的细胞数: {mapped_cells}")
print(f"映射覆盖率: {mapped_cells/total_cells*100:.2f}%")

# Analyze LLM NaN values
print("\n=== LLM NaN Analysis ===\n")
llm_nan_mask = adata.obs['llm_prediction'].isna()
print(f"Total cells with LLM NaN: {llm_nan_mask.sum()}")
print("\nCell types that resulted in LLM NaN:")
for cell_type, count in adata.obs.loc[llm_nan_mask, 'cell_type'].value_counts().items():
    print(f"{cell_type}: {count} cells")

# Print sample of standardized predictions
print("\n=== Sample Predictions ===\n")
sample_cols = ['cell_type', 'cell_type_standard', 'llm_prediction', 'llm_prediction_standard',
               'popv_prediction', 'popv_prediction_standard']
print(adata.obs[sample_cols].head())

# Check unmapped cell types
print("\n=== Unmapped Cell Types ===\n")
unmapped_ref = set(adata.obs['cell_type'].unique()) - set(reference_to_standard.keys())
unmapped_popv = set(adata.obs['popv_prediction'].dropna().unique()) - set(popv_to_standard.keys())

print(f"Unmapped reference types: {len(unmapped_ref)}")
for cell_type in sorted(unmapped_ref):
    count = adata.obs[adata.obs['cell_type'] == cell_type].shape[0]
    print(f"{cell_type}: {count} cells")

print(f"\nUnmapped PopV types: {len(unmapped_popv)}")
for cell_type in sorted(unmapped_popv):
    count = adata.obs[adata.obs['popv_prediction'] == cell_type].shape[0]
    print(f"{cell_type}: {count} cells")

# Calculate match rates
print("\n=== Match Rate Analysis ===\n")

# Calculate exact matches
llm_exact_matches = (adata.obs['cell_type_standard'] == adata.obs['llm_prediction_standard']).sum()
popv_exact_matches = (adata.obs['cell_type_standard'] == adata.obs['popv_prediction_standard']).sum()

# Calculate percentages
llm_match_rate = llm_exact_matches / len(adata) * 100
popv_match_rate = popv_exact_matches / len(adata) * 100

print(f"LLM exact matches: {llm_exact_matches} ({llm_match_rate:.2f}%)")
print(f"PopV exact matches: {popv_exact_matches} ({popv_match_rate:.2f}%)")
print(f"Improvement: {llm_match_rate - popv_match_rate:.2f}%")

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

# For LLM predictions, use the final_consensus column from llm_results
# For PopV predictions, use the popv_prediction column
print("\n=== Prediction Column Analysis ===\n")

# Print information before adding predictions
print("\nBefore adding predictions:")
print(f"PopV results shape: {popv_results.shape}")
print("\nFirst 5 rows of PopV results:")
print(popv_results[['popv_prediction', 'original_celltype']].head())

# Add PopV predictions to adata
adata.obs['popv_prediction'] = popv_results['popv_prediction']
adata.obs['original_celltype'] = popv_results['original_celltype']

# Print information after adding predictions
print("\nAfter adding predictions:")
print("First 5 rows of adata.obs:")
print(adata.obs[['popv_prediction', 'original_celltype']].head())

# Check for NaN values
print("\nNaN check:")
print(f"NaN in popv_prediction: {adata.obs['popv_prediction'].isna().sum()} ({adata.obs['popv_prediction'].isna().mean()*100:.2f}%)")
print(f"NaN in original_celltype: {adata.obs['original_celltype'].isna().sum()} ({adata.obs['original_celltype'].isna().mean()*100:.2f}%)")

# Convert predictions to string type and standardize
adata.obs['popv_prediction'] = adata.obs['popv_prediction'].str.lower().str.strip()
adata.obs['original_celltype'] = adata.obs['original_celltype'].str.lower().str.strip()

# Remove 'nan' strings
adata.obs.loc[adata.obs['popv_prediction'] == 'nan', 'popv_prediction'] = np.nan
adata.obs.loc[adata.obs['original_celltype'] == 'nan', 'original_celltype'] = np.nan

# Print information after standardization
print("\nAfter standardization:")
print("First 5 rows of adata.obs:")
print(adata.obs[['popv_prediction', 'original_celltype']].head())

# Print unique values and their counts
print("\nUnique PopV predictions:")
print(adata.obs['popv_prediction'].value_counts().head(10))
print("\nUnique original celltypes:")
print(adata.obs['original_celltype'].value_counts().head(10))

# Print sample of predictions
print("\nSample of predictions:")
print(adata.obs[['cell_type', 'popv_prediction', 'original_celltype']].head(10))

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
adata.obs['popv_prediction_standard'] = adata.obs['popv_prediction'].map(popv_to_standard)

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

# Define equivalent names
equivalent_names = {
    'NK cells': ['Natural killer cells', 'NK cell', 'Natural killer cell'],
    'T cells': ['T lymphocytes', 'T lymphocyte', 'T-lymphocytes'],
    'B cells': ['B lymphocytes', 'B lymphocyte', 'B-lymphocytes'],
    'Dendritic cells': ['DC', 'DCs', 'Dendritic cell'],
    'Macrophages': ['Macrophage', 'MΦ', 'Mφ'],
    'Monocytes': ['Monocyte'],
    'Endothelial cells': ['Endothelial cell', 'EC', 'ECs'],
    'Epithelial cells': ['Epithelial cell'],
    'Fibroblasts': ['Fibroblast'],
    'Smooth muscle cells': ['Smooth muscle cell', 'SMC', 'SMCs']
}

# Create reverse mapping
reverse_equivalent_names = {}
for ref_type, equiv_list in equivalent_names.items():
    for equiv_type in equiv_list:
        reverse_equivalent_names[equiv_type] = ref_type

def is_equivalent_name(ref_type, pred_type):
    """Check if two cell type names are equivalent"""
    if pd.isna(ref_type) or pd.isna(pred_type):
        return False

    ref_type = str(ref_type)
    pred_type = str(pred_type)

    # Check direct equivalence
    if ref_type in equivalent_names and pred_type in equivalent_names[ref_type]:
        return True

    # Check reverse equivalence
    if pred_type in equivalent_names and ref_type in equivalent_names[pred_type]:
        return True

    # Check through reverse mapping
    if ref_type in reverse_equivalent_names and pred_type == reverse_equivalent_names[ref_type]:
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
    if pd.isna(reference) or pd.isna(prediction):
        return 0.0, "NaN value"

    # Convert to string and normalize
    reference = str(reference)
    prediction = str(prediction)

    # Exact match check (case-insensitive)
    if reference.lower() == prediction.lower():
        return 1.0, "Full match (exact)"

    # Calculate string similarity
    similarity = string_similarity(reference, prediction)
    if 'endothelial' in reference.lower() or 'endothelial' in prediction.lower():
        print(f"Similarity Score: {similarity}")

    # Normalize for comparison
    ref_norm = normalize_cell_type(reference)
    pred_norm = normalize_cell_type(prediction)

    if 'endothelial' in reference.lower() or 'endothelial' in prediction.lower():
        print(f"Normalized Reference: '{ref_norm}'")
        print(f"Normalized Prediction: '{pred_norm}'")

    # Check for equivalent names
    if is_equivalent_name(reference, prediction):
        return 1.0, "Full match (equivalent)"

    # Check for hierarchy relationships
    if 'endothelial' in ref_norm:
        if 'capillary endothelial' in pred_norm or 'arterial endothelial' in pred_norm or 'venous endothelial' in pred_norm:
            return 0.5, "Hierarchy (general to specific: endothelial cells)"

    if 'monocyte' in ref_norm:
        if 'classical monocyte' in pred_norm or 'non-classical monocyte' in pred_norm or 'intermediate monocyte' in pred_norm:
            return 0.5, "Hierarchy (general to specific: monocyte)"

    if 'dendritic' in ref_norm:
        if 'myeloid dendritic' in pred_norm or 'plasmacytoid dendritic' in pred_norm:
            return 0.5, "Hierarchy (general to specific: dendritic cell)"

    if 'epithelial' in ref_norm:
        if any(cell in pred_norm for cell in ['basal', 'club', 'ciliated', 'goblet', 'serous']):
            return 0.5, "Hierarchy (general to specific: epithelial cell)"

    # Check for specific to general relationships
    if 'endothelial cells' == ref_norm and 'endothelial' in pred_norm:
        return 0.5, "Hierarchy (specific to general: endothelial cell)"

    if 'monocyte' == ref_norm and 'monocyte' in pred_norm:
        return 0.5, "Hierarchy (specific to general: monocyte)"

    if 'epithelial cells' == ref_norm and 'epithelial' in pred_norm:
        return 0.5, "Hierarchy (specific to general: epithelial cell)"

    # Check for cell state relationships
    if ('monocyte' in ref_norm and 'macrophage' in pred_norm) or ('macrophage' in ref_norm and 'monocyte' in pred_norm):
        return 0.5, "Cell state (monocyte/macrophage)"

    # No match
    return 0.0, "No match"

# 5. Create detailed comparison DataFrame and analyze matches
comparison_df = pd.DataFrame({
    'Reference': adata.obs['cell_type_standard'],
    'LLM': adata.obs['llm_prediction_standard'],
    'PopV': adata.obs['popv_prediction_standard']
})

# Calculate match scores and types for LLM predictions
llm_matches = []
for idx, row in comparison_df.iterrows():
    score, match_type = calculate_match_score(row['Reference'], row['LLM'])
    llm_matches.append({'score': score, 'type': match_type})

# Add match information to adata
adata.obs['llm_match_score'] = [m['score'] for m in llm_matches]
adata.obs['llm_match_type'] = [m['type'] for m in llm_matches]

# Calculate match scores and types for PopV predictions
popv_matches = []
for idx, row in comparison_df.iterrows():
    score, match_type = calculate_match_score(row['Reference'], row['PopV'])
    popv_matches.append({'score': score, 'type': match_type})

# Add match information to adata
adata.obs['popv_match_score'] = [m['score'] for m in popv_matches]
adata.obs['popv_match_type'] = [m['type'] for m in popv_matches]

# Create DataFrames for mismatches
llm_mismatches = pd.DataFrame({
    'Reference': adata.obs['cell_type_standard'],
    'Prediction': adata.obs['llm_prediction_standard'],
    'Match_Type': adata.obs['llm_match_type']
})

popv_mismatches = pd.DataFrame({
    'Reference': adata.obs['cell_type_standard'],
    'Prediction': adata.obs['popv_prediction_standard'],
    'Match_Type': adata.obs['popv_match_type']
})

# Filter for no matches and partial matches
llm_no_matches = llm_mismatches[llm_mismatches['Match_Type'].str.contains('No match', na=False)]
llm_partial_matches = llm_mismatches[llm_mismatches['Match_Type'].str.contains('Hierarchy|Cell state', na=False)]

# Print match type analysis
print("\n=== Match Type Analysis ===\n")

print("LLMCelltype Match Type Distribution:")
print(adata.obs['llm_match_type'].value_counts())
print(f"\nLLMCelltype Average Score: {adata.obs['llm_match_score'].mean()}")

print("\n=== Detailed Analysis of LLMCelltype Matches ===\n")

# Get all unique no match pairs for LLMCelltype
llm_unique_no_matches = llm_no_matches.groupby(['Reference', 'Prediction']).size().reset_index(name='Count')
llm_unique_no_matches = llm_unique_no_matches.sort_values('Count', ascending=False)

# Get all unique partial match pairs for LLMCelltype
llm_unique_partial_matches = llm_partial_matches.groupby(['Reference', 'Prediction', 'Match_Type']).size().reset_index(name='Count')
llm_unique_partial_matches = llm_unique_partial_matches.sort_values('Count', ascending=False)

# Calculate total cells for each match type
llm_match_type_totals = llm_partial_matches['Match_Type'].value_counts()

# Group partial matches by match type
partial_match_types = llm_unique_partial_matches['Match_Type'].unique()

print("=== No Matches ===\n")
print("Total unique no match pairs for LLMCelltype:", len(llm_unique_no_matches))
print("Total cells with no matches:", len(llm_no_matches))
print("\nTop 20 most frequent LLMCelltype no-match pairs:\n")
for idx, row in llm_unique_no_matches.head(20).iterrows():
    print(f"Reference: '{row['Reference']}'")
    print(f"LLM Prediction: '{row['Prediction']}'")
    print(f"Count: {row['Count']}\n")

print("\n=== Partial Matches ===\n")
print("Total unique partial match pairs for LLMCelltype:", len(llm_unique_partial_matches))
print("Total cells with partial matches:", len(llm_partial_matches))
print("\nBreakdown by Match Type:")
for match_type in llm_match_type_totals.index:
    print(f"{match_type}: {llm_match_type_totals[match_type]} cells")

# Display partial matches grouped by match type
for match_type in partial_match_types:
    matches = llm_unique_partial_matches[llm_unique_partial_matches['Match_Type'] == match_type]
    total_cells = matches['Count'].sum()
    print(f"\n{match_type}:")
    print(f"Total unique pairs: {len(matches)}")
    print(f"Total cells: {total_cells}\n")
    for idx, row in matches.iterrows():
        print(f"Reference: '{row['Reference']}'")
        print(f"LLM Prediction: '{row['Prediction']}'")
        print(f"Count: {row['Count']} ({(row['Count']/total_cells*100):.2f}% of {match_type})\n")

print("\nPopV Match Type Distribution:")
print(adata.obs['popv_match_type'].value_counts())
print(f"\nPopV Average Score: {adata.obs['popv_match_score'].mean()}")

# Continue with original analysis
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
    # Original mappings
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
    'Conventional Dendritic Cells 2 (cDC2)': ['cDC2', 'Type 2 conventional dendritic cells'],

    # AT1/AT2 cell mappings
    'AT1 cells': ['Alveolar Epithelial Type I Cells', 'Type I pneumocytes', 'Alveolar type 1 cells', 'Type 1 alveolar cells', 'pulmonary alveolar type 1 cell'],
    'AT2 cells': ['Alveolar type II pneumocytes', 'Type II pneumocytes', 'Alveolar type 2 cells', 'Type 2 alveolar cells', 'pulmonary alveolar type 2 cell'],
    'Ciliated epithelial cells': ['Multiciliated Cells', 'lung ciliated cell', 'Ciliated cells', 'Ciliated cell'],

    # Endothelial cell mappings
    'Endothelial cells': ['Endothelial cell', 'Blood vessel endothelial cell', 'Vascular endothelial cell'],
    'Capillary endothelial cells': ['Capillary endothelial cell', 'Lung capillary endothelial cell'],
    'Arterial endothelial cells': ['Arterial endothelial cell', 'Artery endothelial cell', 'endothelial cell of artery'],
    'Venous endothelial cells': ['Venous endothelial cell', 'Vein endothelial cell', 'endothelial cell of vein'],
    'Lymphatic endothelial cells': ['Lymphatic endothelial cell', 'endothelial cell of lymphatic vessel', 'Lymph vessel endothelial cell']
}

# Create reverse mapping for equivalent names
reverse_equivalent_names = {}
for ref_type, equiv_list in equivalent_names.items():
    for equiv_type in equiv_list:
        reverse_equivalent_names[equiv_type] = ref_type

def is_equivalent_name(ref_type, pred_type):
    """Check if two cell type names are equivalent"""
    debug_cases = [
        'AT2 cells', 'Alveolar type II pneumocytes',
        'AT1 cells', 'Alveolar Epithelial Type I Cells',
        'Venous endothelial cells', 'Endothelial cell'
    ]

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

    debug_cases = [
        ('AT2 cells', 'Alveolar type II pneumocytes'),
        ('AT1 cells', 'Alveolar Epithelial Type I Cells'),
        ('Venous endothelial cells', 'Endothelial cell')
    ]

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

    # Check for equivalent names with abbreviations and full names
    abbrev_pairs = [
        # Immune cell abbreviations
        ('natural killer cells', 'nk cells'),
        ('gamma delta t cells', 'γδ t cells'),
        ('alpha beta t cells', 'αβ t cells'),

        # AT1/AT2 cell abbreviations and full names
        ('at1 cells', 'alveolar type 1'),
        ('at2 cells', 'alveolar type 2'),
        ('at1 cells', 'type i pneumocytes'),
        ('at2 cells', 'type ii pneumocytes'),
        ('at1 cells', 'alveolar epithelial type i cells'),
        ('at2 cells', 'alveolar epithelial type ii cells'),

        # Endothelial cell abbreviations
        ('endothelial cells', 'ec'),
        ('blood vessel endothelial cells', 'bvec'),
        ('vascular endothelial cells', 'vec'),
        ('capillary endothelial cells', 'capec'),
        ('arterial endothelial cells', 'aec'),
        ('venous endothelial cells', 'vec'),
        ('lymphatic endothelial cells', 'lec'),
        ('pulmonary capillary endothelial cells', 'pcec')
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

    # Define cell type hierarchies based on Cell Ontology (CL) relationships
    cell_hierarchies = {
        # Immune cells
        'immune cell': ['nk cells', 'macrophages', 'monocytes', 'dendritic cells',
                       'neutrophils', 'mast cells', 'plasma cells', 't cells', 'b cells'],
        'monocyte': ['classical monocytes', 'intermediate monocytes', 'non-classical monocytes'],
        'dendritic cell': ['dendritic cells', 'plasmacytoid dendritic cells', 'myeloid dendritic cells'],
        't cell': ['cd4+ t cells', 'cd8+ t cells', 'memory t cells', 'naive t cells',
                  'effector t cells', 'regulatory t cells'],

        # Epithelial cells
        'epithelial cell': ['club cells', 'ciliated epithelial cells', 'basal epithelial cells',
                           'at1 cells', 'at2 cells', 'alveolar epithelial type i cells',
                           'alveolar epithelial type ii cells', 'type i pneumocytes',
                           'type ii pneumocytes', 'multiciliated cells', 'secretory cells',
                           'goblet cells', 'ionocytes', 'neuroendocrine cells'],

        # Endothelial cells
        'endothelial cell': ['capillary endothelial cells', 'arterial endothelial cells',
                            'venous endothelial cells', 'lymphatic endothelial cells',
                            'capillary endothelial cell', 'endothelial cell of artery',
                            'vein endothelial cell', 'endothelial cell of lymphatic vessel',
                            'arterial endothelial cell', 'venous endothelial cell',
                            'lymphatic endothelial cell', 'blood vessel endothelial cell',
                            'vascular endothelial cell'],

        # Stromal cells
        'stromal cell': ['fibroblasts', 'pericytes', 'smooth muscle cells', 'myofibroblasts']
    }

    # Define cell state relationships
    cell_states = [
        ('classical monocyte', 'monocyte', 'Cell state (classical)'),
        ('intermediate monocyte', 'monocyte', 'Cell state (intermediate)'),
        ('non-classical monocyte', 'monocyte', 'Cell state (non-classical)'),
        ('proliferating cell', 'secretory cell', 'Cell state (proliferating)')
    ]

    # Check for cell type hierarchies
    for parent, children in cell_hierarchies.items():
        # Check if one is parent and other is child
        if parent in ref_norm and any(child in pred_norm for child in children):
            return 0.5, f'Hierarchy (general to specific: {parent})'
        if parent in pred_norm and any(child in ref_norm for child in children):
            return 0.5, f'Hierarchy (specific to general: {parent})'

    # Check for cell state relationships
    for state1, state2, desc in cell_states:
        if (state1 in ref_norm and state2 in pred_norm) or \
           (state2 in ref_norm and state1 in pred_norm):
            return 0.5, desc

    return 0.0, "No match"

# Analyze match types for both LLM and PopV predictions
print("\n=== Match Type Analysis ===\n")

# Create DataFrames for mismatches
llm_mismatches = pd.DataFrame({
    'Reference': adata.obs['cell_type_standard'],
    'Prediction': adata.obs['llm_prediction_standard'],
    'Match_Type': adata.obs['llm_match_type']
})

popv_mismatches = pd.DataFrame({
    'Reference': adata.obs['cell_type_standard'],
    'Prediction': adata.obs['popv_prediction_standard'],
    'Match_Type': adata.obs['popv_match_type']
})

# Filter for no matches and partial matches
llm_no_matches = llm_mismatches[llm_mismatches['Match_Type'].str.contains('No match', na=False)]
llm_partial_matches = llm_mismatches[llm_mismatches['Match_Type'].str.contains('Hierarchy|Cell state', na=False)]

# Group and count mismatches
print("LLMCelltype Match Type Distribution:")
print(adata.obs['llm_match_type'].value_counts())
print(f"\nLLMCelltype Average Score: {adata.obs['llm_match_score'].mean()}")

print("\nTop LLMCelltype No Match Pairs:")
print(llm_no_matches.groupby(['Reference', 'Prediction']).size().sort_values(ascending=False).head(20))

print("\nTop LLMCelltype Partial Match Pairs:")
print(llm_partial_matches.groupby(['Reference', 'Prediction', 'Match_Type']).size().sort_values(ascending=False).head(20))

print("\nPopV Match Type Distribution:")
print(adata.obs['popv_match_type'].value_counts())
print(f"\nPopV Average Score: {adata.obs['popv_match_score'].mean()}")

# Original analysis continues
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

# Analyze partial matches in detail
print("\n=== Detailed Analysis of Partial Matches ===\n")

# Get partial matches
llm_partial_matches = llm_match_df[llm_match_df['Score'].between(0.1, 0.9)]

# Group by match type
partial_match_types = llm_partial_matches['Match_Type'].unique()

# Print summary statistics
print("Summary of Partial Matches:")
print(f"Total partial matches: {len(llm_partial_matches)}")
print("\nDistribution by Match Type:")
print(llm_partial_matches['Match_Type'].value_counts())
print(f"\nAverage partial match score: {llm_partial_matches['Score'].mean():.3f}")

# Analyze each match type
for match_type in partial_match_types:
    type_matches = llm_partial_matches[llm_partial_matches['Match_Type'] == match_type]
    print(f"\n{match_type}:")
    print(f"Total cases: {len(type_matches)}")
    print(f"Average score: {type_matches['Score'].mean():.3f}")

    # Get unique pairs and their frequencies
    pairs = pd.DataFrame({
        'Reference': type_matches['Reference'],
        'Prediction': type_matches['Prediction']
    }).value_counts().head(10)

    print("\nTop 10 most frequent pairs:")
    for (ref, pred), count in pairs.items():
        print(f"\nReference: '{ref}'")
        print(f"Prediction: '{pred}'")
        print(f"Count: {count}")

# Print LLM summary statistics
print("\nLLM Match Type Distribution:")
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

# Calculate final scores with detailed analysis for PopV
print("\n=== PopV Detailed Score Analysis ===")
print("PopV match type distribution:")
print(popv_match_df['Match_Type'].value_counts())

print("\nPopV score distribution:")
print(popv_match_df['Score'].value_counts().sort_index())

print("\nPopV match examples for each score:")
for score in sorted(popv_match_df['Score'].unique()):
    sample_matches = popv_match_df[popv_match_df['Score'] == score].head(3)
    print(f"\nScore {score}:")
    for _, row in sample_matches.iterrows():
        print(f"Reference: {row['Reference']} -> Prediction: {row['Prediction']} (Type: {row['Match_Type']})")

# Calculate exact match percentage for PopV
popv_exact_matches = (comparison_df['Reference'] == comparison_df['PopV']).sum()
popv_total = len(comparison_df)
popv_exact_percentage = (popv_exact_matches / popv_total) * 100
print(f"\nPopV Exact Match Percentage: {popv_exact_percentage:.2f}%")

# Calculate weighted accuracy
llm_accuracy = llm_match_df['Score'].mean() * 100

# Calculate PopV weighted accuracy excluding NaN values
popv_valid_scores = popv_match_df[~popv_match_df['Score'].isna()]
popv_accuracy = popv_valid_scores['Score'].mean() * 100
print(f"\nPopV Score Analysis:")
print(f"Total predictions: {len(popv_match_df)}")
print(f"Valid predictions (non-NaN): {len(popv_valid_scores)}")
print(f"PopV Weighted Accuracy (all): {popv_match_df['Score'].mean() * 100:.2f}%")
print(f"PopV Weighted Accuracy (excluding NaN): {popv_accuracy:.2f}%")

improvement_percentage = llm_accuracy - popv_accuracy

# Calculate improvement percentage
improvement_percentage = llm_accuracy - popv_accuracy

# Analyze partial matches in detail
print("\n=== Detailed Analysis of LLMCelltype Matches ===\n")

# Calculate match type distributions
llm_match_type_dist = llm_match_df['Match_Type'].value_counts()
llm_score_bins = pd.cut(llm_match_df['Score'],
                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
llm_score_dist = llm_score_bins.value_counts().sort_index()

# Print match type distribution
print("Match Type Distribution:")
for match_type, count in llm_match_type_dist.items():
    percentage = (count / len(llm_match_df)) * 100
    print(f"{match_type}: {count} ({percentage:.2f}%)")

# Print score distribution
print("\nScore Distribution:")
for score_range, count in llm_score_dist.items():
    percentage = (count / len(llm_match_df)) * 100
    print(f"{score_range}: {count} ({percentage:.2f}%)")

# Analyze partial matches
partial_matches = llm_match_df[llm_match_df['Score'].between(0.1, 0.9)]
print(f"\nPartial Matches Analysis:")
print(f"Total partial matches: {len(partial_matches)} ({(len(partial_matches)/len(llm_match_df))*100:.2f}% of all predictions)")

# Group by match type
for match_type in partial_matches['Match_Type'].unique():
    type_matches = partial_matches[partial_matches['Match_Type'] == match_type]
    print(f"\n{match_type}:")
    print(f"Count: {len(type_matches)} ({(len(type_matches)/len(partial_matches))*100:.2f}% of partial matches)")
    print(f"Average score: {type_matches['Score'].mean():.3f}")

    # Get unique pairs and their frequencies
    pairs = pd.DataFrame({
        'Reference': type_matches['Reference'],
        'Prediction': type_matches['Prediction']
    }).value_counts().head(5)

    print("\nTop 5 most frequent pairs:")
    for (ref, pred), count in pairs.items():
        print(f"Reference: '{ref}' -> Prediction: '{pred}' (Count: {count})")

# Calculate total cells and valid predictions
total_cells = len(comparison_df)

# LLMCelltype calculations
llm_valid = ~(comparison_df['Reference'].isna() | comparison_df['LLM'].isna())
llm_total = llm_valid.sum()
llm_correct = (comparison_df[llm_valid]['Reference'] == comparison_df[llm_valid]['LLM']).sum()
llm_exact_accuracy = (llm_correct / llm_total) * 100 if llm_total > 0 else 0

# PopV calculations
popv_valid = ~(comparison_df['Reference'].isna() | comparison_df['PopV'].isna())
popv_total = popv_valid.sum()
popv_correct = (comparison_df[popv_valid]['Reference'] == comparison_df[popv_valid]['PopV']).sum()
popv_exact_accuracy = (popv_correct / popv_total) * 100 if popv_total > 0 else 0

# Calculate weighted accuracy using match scores
llm_accuracy = llm_match_df['Score'].mean() * 100
popv_accuracy = popv_match_df['Score'].mean() * 100
improvement_percentage = llm_accuracy - popv_accuracy

print(f"\nDetailed Accuracy Results:")
print(f"Total cells in dataset: {total_cells}")

# Calculate different types of matches
strict_exact_matches = (comparison_df[llm_valid]['Reference'] == comparison_df[llm_valid]['LLM']).sum()
singular_plural_matches = llm_match_df[llm_match_df['Match_Type'] == 'Full match (singular/plural variation)'].shape[0]
abbreviation_matches = llm_match_df[llm_match_df['Match_Type'] == 'Full match (abbreviation)'].shape[0]

print(f"\nLLMCelltype Match Types:")
print(f"1. Strict Exact Matches (完全相同字符串):")
print(f"   {strict_exact_matches}/{llm_total} ({strict_exact_matches/llm_total*100:.2f}%)")

print(f"\n2. Full Matches Including Variations:")
print(f"   a. 单复数变化: {singular_plural_matches}/{llm_total} ({singular_plural_matches/llm_total*100:.2f}%)")
print(f"   b. 缩写匹配: {abbreviation_matches}/{llm_total} ({abbreviation_matches/llm_total*100:.2f}%)")

print(f"\n3. Partial Matches (层级关系):")
partial_matches = llm_match_df[llm_match_df['Score'].between(0.1, 0.9)].shape[0]
print(f"   {partial_matches}/{llm_total} ({partial_matches/llm_total*100:.2f}%)")

print(f"\n4. No Matches:")
no_matches = llm_match_df[llm_match_df['Score'] == 0].shape[0]
print(f"   {no_matches}/{llm_total} ({no_matches/llm_total*100:.2f}%)")

print(f"\nOverall Metrics:")
print(f"  - Valid predictions: {llm_total}/{total_cells} cells ({llm_total/total_cells*100:.2f}%)")
print(f"  - Weighted accuracy: {llm_accuracy:.2f}%")
print(f"\nPopV Match Types:")

# 1. 严格匹配
popv_strict_matches = (comparison_df[popv_valid]['Reference'] == comparison_df[popv_valid]['PopV']).sum()
print(f"1. Strict Exact Matches (完全相同字符串):")
print(f"   {popv_strict_matches}/{popv_total} ({popv_strict_matches/popv_total*100:.2f}%)")

print(f"\n2. Full Matches Including Variations:")
# a. 单复数变化
popv_plural_matches = popv_match_df[popv_match_df['Match_Type'] == 'Full match (singular/plural variation)'].shape[0]
print(f"   a. 单复数变化: {popv_plural_matches}/{popv_total} ({popv_plural_matches/popv_total*100:.2f}%)")
# b. 缩写匹配
popv_abbrev_matches = popv_match_df[popv_match_df['Match_Type'] == 'Full match (abbreviation)'].shape[0]
print(f"   b. 缩写匹配: {popv_abbrev_matches}/{popv_total} ({popv_abbrev_matches/popv_total*100:.2f}%)")

# 3. 部分匹配（层级关系）
popv_partial_matches = popv_match_df[popv_match_df['Score'].between(0.1, 0.9)].shape[0]
print(f"\n3. Partial Matches (层级关系):")
print(f"   {popv_partial_matches}/{popv_total} ({popv_partial_matches/popv_total*100:.2f}%)")

# 4. 无匹配
popv_no_matches = popv_match_df[popv_match_df['Score'] == 0].shape[0]
print(f"\n4. No Matches:")
print(f"   {popv_no_matches}/{popv_total} ({popv_no_matches/popv_total*100:.2f}%)")

print(f"\nOverall Metrics:")
print(f"  - Valid predictions: {popv_total}/{total_cells} cells ({popv_total/total_cells*100:.2f}%)")
print(f"  - Weighted accuracy: {popv_accuracy:.2f}%")
print(f"\nImprovement: {improvement_percentage:.2f}%")

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
