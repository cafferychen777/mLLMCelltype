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
# 重新启用adjustText库来避免文本重叠
from adjustText import adjust_text
from color_utils import get_standardized_name, get_cell_type_color

# 设置matplotlib和scanpy参数
plt.rcParams.update({
    'figure.figsize': (30, 6),  # 增加宽度以容纳5个子图
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})
sc.settings.set_figure_params(dpi=300, frameon=True)

# 设置路径
base_dir = "."
output_dir = os.path.join(base_dir, "manuscript/figures")
os.makedirs(output_dir, exist_ok=True)

# 数据路径
data_paths = {
    'thymus': {
        'data': os.path.join(base_dir, "data/raw/Thymus.h5ad"),
        'llm': os.path.join(base_dir, "results/benchmark/popv_comparison/2_evaluation/Thymus_results_fixed.csv"),
        'popv': os.path.join(base_dir, "results/benchmark/popv_comparison/2_evaluation/Thymus_popv_fast_results.csv")
    },
    'lca': {
        'data': os.path.join(base_dir, "data/processed/LCA_with_umap.h5ad"),
        'llm': os.path.join(base_dir, "results/benchmark/reference_comparison/2_evaluation/LCA_results_fixed.csv"),
        'popv': os.path.join(base_dir, "results/benchmark/popv_comparison/2_evaluation/LCA_popv_fast_results.csv")
    }
}

# 加载数据
print("Loading data...")
adata_thymus = sc.read_h5ad(data_paths['thymus']['data'])
llm_results_thymus = pd.read_csv(data_paths['thymus']['llm'])
popv_results_thymus = pd.read_csv(data_paths['thymus']['popv'])

adata_lca = sc.read_h5ad(data_paths['lca']['data'])
llm_results_lca = pd.read_csv(data_paths['lca']['llm'])
popv_results_lca = pd.read_csv(data_paths['lca']['popv'])

# 处理胸腺数据集
print("Processing Thymus data...")

# 创建标准化的映射字典 - 同时标准化键和值
thymus_reference_to_llm_std = {get_standardized_name(k): get_standardized_name(v)
                           for k, v in zip(llm_results_thymus['reference_name'], llm_results_thymus['final_consensus'])}

# 保留原始映射用于向后兼容
thymus_reference_to_llm = {get_standardized_name(k): v for k, v in zip(llm_results_thymus['reference_name'], llm_results_thymus['final_consensus'])}

# 创建细胞级别的PopV预测结果映射
thymus_barcode_to_popv = dict(zip(popv_results_thymus['barcodes'], popv_results_thymus['popv_prediction']))

# 也创建原始细胞类型到最常见PopV注释的映射，用于没有细胞级别映射的情况
thymus_cell_type_to_popv = {}
# 对原始celltype和PopV预测进行标准化 (使用 get_standardized_name)
popv_results_thymus['standardized_celltype'] = popv_results_thymus['original_celltype'].apply(get_standardized_name)
# 按标准化的celltype分组，找出每组中出现频率最高的PopV注释
for celltype, group in popv_results_thymus.groupby('standardized_celltype'):
    popv_counts = group['popv_prediction'].value_counts()
    most_common_popv = popv_counts.idxmax()
    thymus_cell_type_to_popv[celltype] = most_common_popv

# 创建cluster-level PopV注释函数
def get_cluster_level_popv(adata, original_key, popv_key):
    # 存储每个原始细胞类型最常见的PopV注释的字典
    cluster_popv = {}

    # 对于每个原始细胞类型
    for cell_type in adata.obs[original_key].unique():
        # 筛选该原始细胞类型的细胞
        mask = adata.obs[original_key] == cell_type
        # 获取这些细胞的PopV注释
        popv_annotations = adata.obs.loc[mask, popv_key].dropna()

        if len(popv_annotations) > 0:
            # 统计每个PopV注释的出现次数
            popv_counts = popv_annotations.value_counts()
            # 获取最常见的PopV注释
            most_common = popv_counts.idxmax() if len(popv_counts) > 0 else np.nan
            cluster_popv[cell_type] = most_common

    return cluster_popv

# 创建GPT-4o映射字典
thymus_reference_to_gpt4o = {get_standardized_name(k): get_standardized_name(v)
                         for k, v in zip(llm_results_thymus['reference_name'], llm_results_thymus['initial_gpt-4o'])}

# 使用标准化的细胞类型名称进行LLM结果映射 (使用 get_standardized_name)
# 先获取标准化的细胞类型，然后查找标准化的LLM预测结果
adata_thymus.obs['LLMCellType_std'] = adata_thymus.obs['cell_type'].map(
    lambda x: thymus_reference_to_llm_std.get(get_standardized_name(x), 'Unknown'))

# 添加GPT-4o预测结果
adata_thymus.obs['GPT4o_std'] = adata_thymus.obs['cell_type'].map(
    lambda x: thymus_reference_to_gpt4o.get(get_standardized_name(x), 'Unknown'))

# 保留原始的LLMCellType列以便向后兼容
adata_thymus.obs['LLMCellType'] = adata_thymus.obs['cell_type'].map(
    lambda x: thymus_reference_to_llm.get(get_standardized_name(x), 'Unknown'))

# 使用细胞级别的PopV预测结果
# 首先尝试使用barcode进行映射
adata_thymus.obs['PopV'] = adata_thymus.obs.index.map(lambda x: thymus_barcode_to_popv.get(x, None))
# 对于没有直接映射的细胞，使用原始细胞类型的最常见PopV注释 (使用 get_standardized_name)
adata_thymus.obs['PopV'] = adata_thymus.obs['PopV'].fillna(adata_thymus.obs['cell_type'].map(lambda x: thymus_cell_type_to_popv.get(get_standardized_name(x), 'Unknown')))

# 创建胸腺数据集的cluster-level PopV注释
thymus_cluster_popv = get_cluster_level_popv(adata_thymus, 'cell_type', 'PopV')
adata_thymus.obs['ClusterPopV'] = adata_thymus.obs['cell_type'].map(thymus_cluster_popv)

# 处理LCA数据集
print("Processing LCA data...")

# 创建映射字典 - 同时标准化键和值
lca_reference_to_llm_std = {get_standardized_name(k): get_standardized_name(v)
                        for k, v in zip(llm_results_lca['reference_name'], llm_results_lca['final_consensus'])}

# 保留原始映射用于向后兼容
lca_reference_to_llm = dict(zip(llm_results_lca['reference_name'], llm_results_lca['final_consensus']))

# 应用名称标准化 (使用color_utils中的get_standardized_name函数)
adata_lca.obs['cell_type_norm'] = adata_lca.obs['cell_type'].apply(get_standardized_name)

# 创建GPT-4o映射字典
lca_reference_to_gpt4o = {get_standardized_name(k): get_standardized_name(v)
                       for k, v in zip(llm_results_lca['reference_name'], llm_results_lca['initial_gpt-4o'])}

# 获取标准化的LLM预测结果
lca_llm_std = [lca_reference_to_llm_std.get(get_standardized_name(x), "Unknown") for x in adata_lca.obs['cell_type']]
adata_lca.obs['LLMCellType_std'] = lca_llm_std

# 获取标准化的GPT-4o预测结果
lca_gpt4o_std = [lca_reference_to_gpt4o.get(get_standardized_name(x), "Unknown") for x in adata_lca.obs['cell_type']]
adata_lca.obs['GPT4o_std'] = lca_gpt4o_std

# 保留原始的final_consensus_norm列以便向后兼容
lca_consensus_norm = [lca_reference_to_llm.get(get_standardized_name(x), "Unknown") for x in adata_lca.obs['cell_type']]
adata_lca.obs['final_consensus_norm'] = lca_consensus_norm

# 处理PopV预测结果
popv_results_lca['popv_prediction_norm'] = popv_results_lca['popv_prediction']
adata_lca.obs['popv_prediction_norm'] = popv_results_lca.set_index('index')['popv_prediction_norm']

# 创建LCA数据集的cluster-level PopV注释
lca_cluster_popv = get_cluster_level_popv(adata_lca, 'cell_type_norm', 'popv_prediction_norm')
adata_lca.obs['cluster_popv_standard'] = adata_lca.obs['cell_type_norm'].map(lca_cluster_popv)



# Function to get standardized label
def calculate_centroids(coords, labels):
    """Calculate centroids for each unique label in the data.

    Args:
        coords: np.array of coordinates
        labels: np.array of labels
    Returns:
        dict: Mapping of labels to their centroids
    """
    unique_labels = np.unique(labels)
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            centroid = np.mean(coords[mask], axis=0)
            centroids[label] = centroid
    return centroids

# 设置绘图样式
sc.settings.set_figure_params(dpi=300, frameon=False)
plt.rcParams['figure.figsize'] = (18, 12)  # 2*3布局的大小
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True

# 创建2*3的子图布局
print("Plotting UMAPs...")
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

# 设置子图间距
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 设置所有子图的样式
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_aspect('equal', adjustable='box')
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

# 添加胸腺数据集的标题（上排）
ax1.text(0.5, 1.02, 'Thymus Reference', transform=ax1.transAxes, fontsize=14, ha='center')
ax2.text(0.5, 1.02, 'Thymus LLMCellType', transform=ax2.transAxes, fontsize=14, ha='center')
ax3.text(0.5, 1.02, 'Thymus PopV', transform=ax3.transAxes, fontsize=14, ha='center')

# 添加LCA数据集的标题（下排）
ax4.text(0.5, 1.02, 'LCA Reference', transform=ax4.transAxes, fontsize=14, ha='center')
ax5.text(0.5, 1.02, 'LCA LLMCellType', transform=ax5.transAxes, fontsize=14, ha='center')
ax6.text(0.5, 1.02, 'LCA PopV', transform=ax6.transAxes, fontsize=14, ha='center')

# 获取UMAP坐标
thymus_umap_coords = adata_thymus.obsm['X_umap']
lca_umap_coords = adata_lca.obsm['X_umap']


# Nature Methods color palette
# Using a professional, color-blind friendly palette
# Main categories have distinct base colors, subcategories use different shade

# 获取UMAP坐标
thymus_umap_coords = adata_thymus.obsm['X_umap']
lca_umap_coords = adata_lca.obsm['X_umap']


def plot_umap_annotation(ax, adata, annotation_key, umap_coords):
    """Plot UMAP with cell type annotations.

    Args:
        ax: matplotlib axis
        adata: AnnData object
        annotation_key: Key for cell type annotations
        umap_coords: UMAP coordinates
    """
    # 计算坐标范围
    x_min, x_max = np.min(umap_coords[:, 0]), np.max(umap_coords[:, 0])
    y_min, y_max = np.min(umap_coords[:, 1]), np.max(umap_coords[:, 1])

    # 确保正方形显示区域
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    # 设置相等的坐标轴范围
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)

    texts = []
    # 首先创建一个字典来存储每个标准化名称的所有点
    standardized_groups = {}
    for ct in np.unique(adata.obs[annotation_key]):
        standardized_name = get_standardized_name(ct)
        mask = adata.obs[annotation_key] == ct
        print(f"Cell type: {ct} -> {standardized_name}")
        color = get_cell_type_color(standardized_name)
        print(f"Color: {color}")
        # 绘制散点
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                 c=[color], s=0.2, alpha=0.3, rasterized=True)

        # 将当前mask对应的点添加到对应的标准化名称组中
        if standardized_name not in standardized_groups:
            standardized_groups[standardized_name] = mask
        else:
            standardized_groups[standardized_name] |= mask

    # 为每个标准化名称添加文本标签
    for standardized_name, combined_mask in standardized_groups.items():
        if np.sum(combined_mask) > 0:
            # 计算所有属于这个标准化名称的点的中心
            centroid = np.mean(umap_coords[combined_mask], axis=0)

            # 创建文本对象，使用scatter来标记中心点
            color = get_cell_type_color(standardized_name)
            ax.scatter(centroid[0], centroid[1], c=color, s=10, alpha=0.8, edgecolors='black', linewidths=0.5, rasterized=True)

            # 添加文本标签
            text = ax.text(centroid[0], centroid[1], standardized_name,
                      fontsize=8, ha='center', va='center',
                      color='black')

            # 将文本对象添加到texts列表中，用于后续的adjust_text调整
            texts.append(text)

    # 设置相等的纵横比
    ax.set_aspect('equal', adjustable='box')
    # 使用adjust_text来调整文本位置
    adjust_text(texts, ax=ax,
                force_points=(0.2, 0.4),
                force_text=(1.0, 2.0),
                expand_points=(2.0, 2.0),
                expand_text=(2.0, 2.0),
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5, alpha=0.7)
                )

# 获取UMAP坐标
thymus_umap_coords = adata_thymus.obsm['X_umap']
lca_umap_coords = adata_lca.obsm['X_umap']

# 创建2*4的子图布局
print("Plotting UMAPs...")
fig = plt.figure(figsize=(24, 12))

# 计算子图之间的间距
wspace = 0.3  # 水平间距
hspace = 0.3  # 垂直间距

# 创建GridSpec，指定间距
gs = GridSpec(2, 4, figure=fig, wspace=wspace, hspace=hspace)

# 创建子图和设置标题
axes = []
titles = [
    'Reference (Thymus)', 'Our Framework (Thymus)', 'GPTCelltype (Thymus)', 'Cluster-Level PopV (Thymus)',
    'Reference (LCA)', 'Our Framework (LCA)', 'GPTCelltype (LCA)', 'Cluster-Level PopV (LCA)'
]

for i in range(8):  # 现在有8个子图
    ax = fig.add_subplot(gs[i//4, i%4])
    ax.set_title(titles[i], fontsize=14, pad=5)  # 增加标题和图之间的距离
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    axes.append(ax)

# 绘制胸腺数据集（上排）
plot_umap_annotation(axes[0], adata_thymus, 'cell_type', thymus_umap_coords)
plot_umap_annotation(axes[1], adata_thymus, 'LLMCellType_std', thymus_umap_coords)  # 使用标准化的LLM结果
plot_umap_annotation(axes[2], adata_thymus, 'GPT4o_std', thymus_umap_coords)  # GPT-4o结果
plot_umap_annotation(axes[3], adata_thymus, 'ClusterPopV', thymus_umap_coords)  # 集群级别的PopV

# 绘制LCA数据集（下排）
plot_umap_annotation(axes[4], adata_lca, 'cell_type_norm', lca_umap_coords)
plot_umap_annotation(axes[5], adata_lca, 'LLMCellType_std', lca_umap_coords)  # 使用标准化的LLM结果
plot_umap_annotation(axes[6], adata_lca, 'GPT4o_std', lca_umap_coords)  # GPT-4o结果
plot_umap_annotation(axes[7], adata_lca, 'cluster_popv_standard', lca_umap_coords)  # 集群级别的PopV

# 调整布局并保存图像
plt.tight_layout()

# 保存完整图像
for fmt in ['pdf', 'png']:
    plt.savefig(
        os.path.join(base_dir, f'manuscript/figures/panel_bc.{fmt}'),
        dpi=300, format=fmt, bbox_inches='tight'
    )

# 保存上半部分 (Thymus)
for ax in axes[4:]:  # 隐藏下半部分
    ax.set_visible(False)
for fmt in ['pdf', 'png']:
    plt.savefig(
        os.path.join(base_dir, f'manuscript/figures/panel_b.{fmt}'),
        dpi=300, format=fmt, bbox_inches='tight'
    )

# 恢复上半部分的可见性并隐藏下半部分
for ax in axes[4:]:  # 恢复下半部分
    ax.set_visible(True)
for ax in axes[:4]:  # 隐藏上半部分
    ax.set_visible(False)

# 保存下半部分 (PBMC)
for fmt in ['pdf', 'png']:
    plt.savefig(
        os.path.join(base_dir, f'manuscript/figures/panel_c.{fmt}'),
        dpi=300, format=fmt, bbox_inches='tight'
    )

plt.close()

print("Figure 4 panels B,C saved successfully.")
