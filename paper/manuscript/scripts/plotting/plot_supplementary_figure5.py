#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np
from adjustText import adjust_text

# 设置matplotlib参数
plt.rcParams['figure.figsize'] = (14, 14)  # 改为方形
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 设置scanpy参数
sc.settings.set_figure_params(dpi=300, frameon=True)

# 设置路径
base_dir = "."
schiller_data_path = os.path.join(base_dir, "data/raw/Schiller_2020.h5ad")
schiller_results_path = os.path.join(base_dir, "results/benchmark/reference_comparison/2_evaluation/Schiller_2020_results.csv")
sun_data_path = os.path.join(base_dir, "data/raw/Sun_2020.h5ad")
sun_results_path = os.path.join(base_dir, "results/benchmark/reference_comparison/2_evaluation/sun_2020_results.csv")
output_dir = os.path.join(base_dir, "manuscript/figures")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 定义标准化名称映射
standardized_labels = {
    # 上皮细胞
    'AT1': 'Alveolar type 1 cells (AT1)',
    'AT2': 'Alveolar type 2 cells (AT2)',
    'Alveolar Type 1 cells': 'Alveolar type 1 cells (AT1)',
    'Alveolar epithelial type II cells (AT2)': 'Alveolar type 2 cells (AT2)',
    'Alveolar Epithelial Type II (AT2)': 'Alveolar type 2 cells (AT2)',
    'Basal': 'Basal epithelial cells',
    'Basal cells': 'Basal epithelial cells',
    'Epithelial cells': 'Basal epithelial cells',
    'Secretory': 'Club cells',
    'Club Cells': 'Club cells',
    'Multiciliated lineage': 'Ciliated epithelial cells',
    'Ciliated cells': 'Ciliated epithelial cells',
    'Submucosal Secretory': 'Submucosal secretory cells',

    # T细胞和NK细胞
    'T cell lineage': 'T cells',
    'Immune T cells': 'T cells',
    'T cells': 'T cells',
    'Innate lymphoid cell NK': 'NK cells',
    'Cytotoxic T cells/NK cells': 'NK cells',

    # B细胞
    'B cell lineage': 'B cells',
    'Plasma B cells': 'B cells',
    'B cells': 'B cells',

    # 巨噬细胞和单核细胞
    'Macrophages': 'Alveolar macrophages',
    'Alveolar Macrophages': 'Alveolar macrophages',
    'Monocytes': 'Monocytes',
    'Monocytes and Macrophages': 'Monocytes',
    'Dendritic cells': 'Dendritic cells',

    # 内皮细胞
    'EC venous': 'Venous endothelial cells',
    'Endothelial cells': 'Venous endothelial cells',  # 改回原始映射
    'EC arterial': 'Arterial endothelial cells',
    'Vascular endothelial cells': 'Arterial endothelial cells',
    'EC capillary': 'Capillary endothelial cells',
    'Lymphatic EC mature': 'Lymphatic endothelial cells',
    'Lymphatic EC differentiating': 'Lymphatic endothelial cells',

    # 基质细胞
    'Fibroblasts': 'Fibroblasts',
    'Myofibroblasts': 'Myofibroblasts',
    'SM activated stress response': 'Smooth muscle cells',
    'Smooth muscle cells': 'Smooth muscle cells',

    # 其他细胞
    'Mast cells': 'Mast cells',
    'Rare': 'Rare cells',  # 保持原始名称
    'Neuroendocrine cells': 'Neuroendocrine cells'  # 保持原始名称
}

# 定义颜色方案（使用标准化后的名称）
color_scheme = {
    # 上皮细胞 (绿色系)
    'Alveolar type 1 cells (AT1)': '#006D2C',  # 深绿
    'Alveolar type 2 cells (AT2)': '#238B45',  # 中绿
    'Club cells': '#41AE76',  # 浅绿
    'Ciliated epithelial cells': '#66C2A4',  # 更浅绿
    'Basal epithelial cells': '#99D8C9',  # 最浅绿
    'Submucosal secretory cells': '#B7E2D3',  # 最最浅绿

    # T细胞和NK细胞 (蓝色系)
    'T cells': '#08519C',  # 深蓝
    'NK cells': '#3182BD',  # 中蓝

    # B细胞 (紫色系)
    'B cells': '#88419D',  # 深紫

    # 巨噬细胞和单核细胞 (棕色系)
    'Alveolar macrophages': '#8C510A',  # 深棕
    'Monocytes': '#BF812D',  # 中棕
    'Dendritic cells': '#DFC27D',  # 浅棕

    # 内皮细胞 (红色系)
    'Venous endothelial cells': '#CB181D',  # 深红
    'Arterial endothelial cells': '#EF3B2C',  # 中红
    'Capillary endothelial cells': '#FB6A4A',  # 浅红
    'Lymphatic endothelial cells': '#FC9272',  # 更浅红

    # 基质细胞 (橙色系)
    'Fibroblasts': '#E6550D',  # 深橙
    'Myofibroblasts': '#FD8D3C',  # 中橙
    'Smooth muscle cells': '#FDAE6B',  # 浅橙

    # 其他细胞
    'Mast cells': '#DE77AE',  # 粉红
    'Rare cells': '#C7C7C7',  # 浅灰
    'Neuroendocrine cells': '#969696',  # 中灰

    # 未知细胞
    'Unknown': '#808080',  # 灰色
    'nan': '#808080'  # 灰色
}

def process_dataset(data_path, results_path):
    """处理单个数据集"""
    # 读取数据
    adata = sc.read_h5ad(data_path)
    llm_results = pd.read_csv(results_path)

    # 将category转换为字符串并统一命名
    adata.obs['ann_level_3'] = adata.obs['ann_level_3'].astype(str)

    # 创建从原始注释到LLM注释的映射
    mapping_dict = dict(zip(llm_results['reference_name'], llm_results['final_consensus']))

    # 添加LLM注释
    adata.obs['LLMCellType'] = adata.obs['ann_level_3'].map(lambda x: mapping_dict.get(x, x))

    return adata

def plot_umap(adata, ax1, ax2, label1, label2):
    """绘制单个数据集的UMAP图"""
    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']

    texts1 = []
    for ct in sorted(adata.obs['ann_level_3'].unique()):
        if ct not in ['Unknown', 'nan']:
            mask = adata.obs['ann_level_3'] == ct
            standardized_label = standardized_labels.get(ct, ct)
            color = color_scheme.get(standardized_label, '#D3D3D3')
            if color == '#D3D3D3':
                print(f"Warning: No color defined for reference cell type: {standardized_label}")
            ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                        c=[color], s=20, alpha=0.7, label=standardized_label, rasterized=True)
            if np.sum(mask) > 0:
                centroid = np.mean(umap_coords[mask], axis=0)
                texts1.append(ax1.text(centroid[0], centroid[1], standardized_label,
                         fontsize=10, ha='center', va='center'))

    texts2 = []
    for ct in sorted(adata.obs['LLMCellType'].unique()):
        if pd.notna(ct) and ct != 'Unknown':  # 移除去重逻辑，与原始脚本保持一致
            mask = adata.obs['LLMCellType'] == ct
            standardized_label = standardized_labels.get(ct, ct)
            color = color_scheme.get(standardized_label, '#D3D3D3')
            if color == '#D3D3D3':
                print(f"Warning: No color defined for LLM cell type: {standardized_label}")
            ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                        c=[color], s=20, alpha=0.7, label=standardized_label, rasterized=True)
            if np.sum(mask) > 0:
                centroid = np.mean(umap_coords[mask], axis=0)
                texts2.append(ax2.text(centroid[0], centroid[1], standardized_label,
                         fontsize=10, ha='center', va='center'))

    # Adjust text labels
    adjust_text(texts1, ax=ax1, force_points=0.5, force_text=0.8, expand_points=(2, 2))
    adjust_text(texts2, ax=ax2, force_points=0.5, force_text=0.8, expand_points=(2, 2))

    # Set axis limits to be the same for both plots
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    # Add subplot labels and titles
    ax1.text(-0.1, 1.1, label1, transform=ax1.transAxes, fontsize=18, weight='bold')
    ax2.text(-0.1, 1.1, label2, transform=ax2.transAxes, fontsize=18, weight='bold')

    # Add titles
    ax1.text(0.5, 1.02, 'Reference', transform=ax1.transAxes, fontsize=16, ha='center')
    ax2.text(0.5, 1.02, 'Our Framework', transform=ax2.transAxes, fontsize=16, ha='center')

    # Add UMAP axis labels
    ax1.set_xlabel('UMAP1', fontsize=14)
    ax1.set_ylabel('UMAP2', fontsize=14)
    ax2.set_xlabel('UMAP1', fontsize=14)
    ax2.set_ylabel('UMAP2', fontsize=14)

    # Remove ticks but keep axis labels and borders
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

# 读取和处理数据
print("Reading Schiller 2020 data...")
schiller_adata = process_dataset(schiller_data_path, schiller_results_path)

print("Reading Sun 2020 data...")
sun_adata = process_dataset(sun_data_path, sun_results_path)

# 创建画布
print("Plotting UMAPs...")
fig = plt.figure(figsize=(14, 14))
gs = GridSpec(2, 2, figure=fig)  # 改为2x2布局

# Schiller 2020 plots (上排)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
plot_umap(schiller_adata, ax1, ax2, 'a', 'b')

# Sun 2020 plots (下排)
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
plot_umap(sun_adata, ax3, ax4, 'c', 'd')

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'supplementary_figure5.pdf'),
            bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, 'supplementary_figure5.png'),
            bbox_inches='tight', dpi=300)
plt.close()

print("Done! Plots saved in:", output_dir)
