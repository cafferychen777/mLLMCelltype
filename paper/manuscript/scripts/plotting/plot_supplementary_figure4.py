#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# 设置matplotlib参数
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 设置scanpy参数
sc.settings.set_figure_params(dpi=600, frameon=True)

# 设置路径
base_dir = "."
data_path = os.path.join(base_dir, "data/raw/HLCA_Core.h5ad")
results_path = os.path.join(base_dir, "results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_3_results.csv")
output_dir = os.path.join(base_dir, "manuscript/figures")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取数据
print("Reading data...")
adata = sc.read_h5ad(data_path)
results_df = pd.read_csv(results_path)

# 创建映射字典
print("Creating mapping dictionaries...")
ann_level3_dict = dict(zip(results_df['reference_name'], results_df['ann_level_3']))
aligned_level3_dict = dict(zip(results_df['reference_name'], results_df['aligned_level3']))

manual_dict = {
    'Lymphatic EC differentiating': 'Lymphatic EC',
    'Lymphatic EC proliferating': 'Lymphatic EC',
    'Lymphatic EC mature': 'Lymphatic EC'
}

# 应用映射
print("Adding annotations...")
adata.obs['Reference'] = adata.obs['leiden_3'].map(ann_level3_dict).fillna('Unknown')
adata.obs['LLMCellType'] = adata.obs['leiden_3'].map(aligned_level3_dict).fillna('Unknown')

adata.obs['Reference'] = adata.obs['Reference'].map(manual_dict).fillna(adata.obs['Reference'])
adata.obs['LLMCellType'] = adata.obs['LLMCellType'].map(manual_dict).fillna(adata.obs['LLMCellType'])

# 清理Fibroblasts的命名问题
adata.obs['LLMCellType'] = adata.obs['LLMCellType'].str.strip()

# 定义统一的颜色方案
color_scheme = {
    # Immune cells (blue spectrum)
    'Innate lymphoid cell NK': '#2166AC',
    'Macrophages': '#4393C3',
    'Monocytes': '#92C5DE',
    'B cell lineage': '#084594',
    'Dendritic cells': '#8C96C6',
    'Mast cells': '#8856A7',
    'T cell lineage': '#54278F',

    # Epithelial cells (green spectrum)
    'AT1': '#238B45',
    'AT2': '#41AE76',
    'Basal': '#006D2C',
    'Secretory': '#74C476',
    'Multiciliated lineage': '#99D8C9',
    'Ciliated cells': '#99D8C9',
    'Submucosal Secretory': '#66C2A4',
    'Salivary gland epithelial cells': '#66C2A4',
    'Stress-responsive epithelial cells': '#A1D99B',
    'Suprabasal Epithelial Cells': '#BAE4B3',

    # Endothelial cells (red spectrum)
    'EC capillary': '#FB6A4A',
    'EC arterial': '#EF3B2C',
    'EC venous': '#FB6A4A',
    'Lymphatic EC': '#FC9272',
    'Capillary Arterial Endothelial cells': '#EF3B2C',

    # Stromal cells (brown spectrum)
    'Fibroblasts': '#8C510A',
    'SM activated stress response': '#BF812D',
    'Myofibroblasts': '#DFC27D',

    # Other
    'Rare': '#C51B7D',
    'Unknown': '#808080'
}

# 创建UMAP图
print("Plotting UMAPs...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# 添加子图标签
ax1.text(-0.1, 1.1, 'a', transform=ax1.transAxes, fontsize=24, fontweight='bold')
ax2.text(-0.1, 1.1, 'b', transform=ax2.transAxes, fontsize=24, fontweight='bold')

# 设置栅格化 (在 axes 级别)
ax1.set_rasterization_zorder(1)
ax2.set_rasterization_zorder(1)

# Plot Reference annotations (use normal font weight, consistent with supplementary_figure8)
sc.pl.umap(adata, color='Reference', ax=ax1, show=False,
           palette=color_scheme, size=50, legend_loc='on data',
           legend_fontsize=8, legend_fontweight='normal')

# Plot LLMCellType annotations
sc.pl.umap(adata, color='LLMCellType', ax=ax2, show=False,
           palette=color_scheme, size=50, legend_loc='on data',
           legend_fontsize=8, legend_fontweight='normal')

# 手动设置标题和轴标签
ax1.set_title('Reference', fontsize=20)
ax2.set_title('Our Framework', fontsize=20)

# 设置轴标签
ax1.set_xlabel('UMAP1', fontsize=16)
ax1.set_ylabel('UMAP2', fontsize=16)
ax2.set_xlabel('UMAP1', fontsize=16)
ax2.set_ylabel('UMAP2', fontsize=16)

# 设置刻度标签字体大小
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, 'supplementary_figure4')
plt.savefig(f"{output_path}.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(f"{output_path}.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nVisualization saved to:")
print(f"{output_path}.pdf")
print(f"{output_path}.png")

# Print statistics
print("\n=== Cell Type Statistics ===")
print("\nNumber of cells per type in Reference:")
print(adata.obs['Reference'].value_counts())
print("\nNumber of cells per type in LLMCellType:")
print(adata.obs['LLMCellType'].value_counts())
