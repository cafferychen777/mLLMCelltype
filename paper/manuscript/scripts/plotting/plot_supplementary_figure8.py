#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Figure 8: popV comparison across benchmark datasets
Layout:
  - Panel a: Bar chart comparing mLLMCelltype vs popV (7 datasets)
  - Panel b: Thymus popV UMAP
  - Panel c: LCA popV UMAP
"""

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.gridspec import GridSpec
from adjustText import adjust_text
from color_utils import get_standardized_name, get_cell_type_color

# 设置matplotlib参数
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
})
sc.settings.set_figure_params(dpi=300, frameon=True)

# 设置路径
base_dir = "."
output_dir = os.path.join(base_dir, "manuscript/figures")
os.makedirs(output_dir, exist_ok=True)

# ============================================
# Panel A: Bar chart 数据
# ============================================
comparison_data = {
    'Dataset': ['Thymus', 'LCA', 'GTEx\nLung', 'GTEx\nHeart',
                'Azimuth\nPancreas', 'Azimuth\nKidney', 'Azimuth\nLiver'],
    'mLLMCelltype': [59.1, 58.7, 89.1, 91.7, 69.2, 59.4, 73.9],
    'popV': [48.5, 45.5, 79.0, 82.0, 58.0, 48.0, 62.0],
    'scANVI': [28.3, 61.4, 80.8, 79.1, 62.2, 25.6, 59.6],
    'Category': ['Independent', 'Independent', 'GTEx', 'GTEx',
                 'Azimuth', 'Azimuth', 'Azimuth']
}
df_comparison = pd.DataFrame(comparison_data)

# ============================================
# 数据路径 (UMAP)
# ============================================
data_paths = {
    'thymus': {
        'data': os.path.join(base_dir, "data/raw/Thymus.h5ad"),
        'popv': os.path.join(base_dir, "results/benchmark/popv_comparison/2_evaluation/Thymus_popv_fast_results.csv")
    },
    'lca': {
        'data': os.path.join(base_dir, "data/processed/LCA_with_umap.h5ad"),
        'popv': os.path.join(base_dir, "results/benchmark/popv_comparison/2_evaluation/LCA_popv_fast_results.csv")
    }
}

# 加载数据
print("Loading data...")
adata_thymus = sc.read_h5ad(data_paths['thymus']['data'])
popv_results_thymus = pd.read_csv(data_paths['thymus']['popv'])

adata_lca = sc.read_h5ad(data_paths['lca']['data'])
popv_results_lca = pd.read_csv(data_paths['lca']['popv'])

# 处理胸腺数据集的PopV预测
print("Processing Thymus PopV data...")
thymus_barcode_to_popv = dict(zip(popv_results_thymus['barcodes'],
                                   popv_results_thymus['popv_prediction']))
adata_thymus.obs['PopV'] = adata_thymus.obs.index.map(
    lambda x: thymus_barcode_to_popv.get(x, None))

popv_results_thymus['standardized_celltype'] = popv_results_thymus['original_celltype'].apply(
    get_standardized_name)
thymus_cell_type_to_popv = {}
for celltype, group in popv_results_thymus.groupby('standardized_celltype'):
    popv_counts = group['popv_prediction'].value_counts()
    most_common_popv = popv_counts.idxmax()
    thymus_cell_type_to_popv[celltype] = most_common_popv

adata_thymus.obs['PopV'] = adata_thymus.obs['PopV'].fillna(
    adata_thymus.obs['cell_type'].map(
        lambda x: thymus_cell_type_to_popv.get(get_standardized_name(x), 'Unknown')))

# 处理LCA数据集的PopV预测
print("Processing LCA PopV data...")
popv_results_lca['popv_prediction_norm'] = popv_results_lca['popv_prediction']
adata_lca.obs['popv_prediction_norm'] = popv_results_lca.set_index('index')['popv_prediction_norm']

# 获取UMAP坐标
thymus_umap_coords = adata_thymus.obsm['X_umap']
lca_umap_coords = adata_lca.obsm['X_umap']


def plot_umap_annotation(ax, adata, annotation_key, umap_coords, title):
    """Plot UMAP with cell type annotations."""
    # 计算坐标范围
    x_min, x_max = np.min(umap_coords[:, 0]), np.max(umap_coords[:, 0])
    y_min, y_max = np.min(umap_coords[:, 1]), np.max(umap_coords[:, 1])

    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    ax.set_xlim(x_center - max_range/2 * 1.1, x_center + max_range/2 * 1.1)
    ax.set_ylim(y_center - max_range/2 * 1.1, y_center + max_range/2 * 1.1)

    texts = []
    standardized_groups = {}

    for ct in np.unique(adata.obs[annotation_key].dropna()):
        standardized_name = get_standardized_name(ct)
        mask = adata.obs[annotation_key] == ct
        color = get_cell_type_color(standardized_name)

        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=[color], s=0.3, alpha=0.4, rasterized=True)

        if standardized_name not in standardized_groups:
            standardized_groups[standardized_name] = mask
        else:
            standardized_groups[standardized_name] |= mask

    for standardized_name, combined_mask in standardized_groups.items():
        if np.sum(combined_mask) > 0:
            centroid = np.mean(umap_coords[combined_mask], axis=0)
            color = get_cell_type_color(standardized_name)
            ax.scatter(centroid[0], centroid[1], c=color, s=12, alpha=0.9,
                       edgecolors='black', linewidths=0.5, rasterized=True)

            text = ax.text(centroid[0], centroid[1], standardized_name,
                           fontsize=7, ha='center', va='center', color='black')
            texts.append(text)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel('UMAP1', fontsize=9)
    ax.set_ylabel('UMAP2', fontsize=9)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.grid(False)

    adjust_text(texts, ax=ax,
                force_points=(0.3, 0.5),
                force_text=(0.8, 1.5),
                expand_points=(1.5, 1.5),
                expand_text=(1.5, 1.5),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, alpha=0.6))


# ============================================
# 创建图形: 2行布局 (bar chart + 2 UMAPs)
# ============================================
print("Creating Extended Figure 8...")

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1], wspace=0.25, hspace=0.3)

# 颜色定义 (ggsci NPG style)
COLOR_MLLM = '#E64B35'  # NPG red
COLOR_POPV = '#4DBBD5'  # NPG cyan
COLOR_SCANVI = '#00A087'  # NPG green

# ============================================
# Panel A: Bar chart (占据第一行两列)
# ============================================
ax_bar = fig.add_subplot(gs[0, :])

x = np.arange(len(df_comparison))
width = 0.25

bars1 = ax_bar.bar(x - width, df_comparison['mLLMCelltype'], width,
                   label='mLLMCelltype', color=COLOR_MLLM)
bars2 = ax_bar.bar(x, df_comparison['popV'], width,
                   label='popV', color=COLOR_POPV)
bars3 = ax_bar.bar(x + width, df_comparison['scANVI'], width,
                   label='scANVI', color=COLOR_SCANVI)

# 格式设置
ax_bar.set_ylabel('Annotation Accuracy (%)', fontsize=11)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(df_comparison['Dataset'], fontsize=9)
ax_bar.set_ylim(0, 100)
ax_bar.legend(loc='upper left', frameon=False, fontsize=9)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

# Panel 标签
ax_bar.text(-0.05, 1.05, 'a', transform=ax_bar.transAxes,
            fontsize=16, fontweight='bold')

# ============================================
# Panel B: Thymus UMAP
# ============================================
ax_thymus = fig.add_subplot(gs[1, 0])
plot_umap_annotation(ax_thymus, adata_thymus, 'PopV', thymus_umap_coords,
                     'popV predictions (Thymus)')
ax_thymus.text(-0.08, 1.02, 'b', transform=ax_thymus.transAxes,
               fontsize=16, fontweight='bold')

# ============================================
# Panel C: LCA UMAP
# ============================================
ax_lca = fig.add_subplot(gs[1, 1])
plot_umap_annotation(ax_lca, adata_lca, 'popv_prediction_norm', lca_umap_coords,
                     'popV predictions (LCA)')
ax_lca.text(-0.08, 1.02, 'c', transform=ax_lca.transAxes,
            fontsize=16, fontweight='bold')

# ============================================
# 保存图像
# ============================================
plt.tight_layout()

for fmt in ['pdf', 'png']:
    output_path = os.path.join(output_dir, f'supplementary_figure8.{fmt}')
    plt.savefig(output_path, dpi=300, format=fmt, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

plt.close()
print("Extended Figure 8 completed!")
