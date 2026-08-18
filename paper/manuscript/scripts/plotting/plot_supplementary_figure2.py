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
plt.rcParams['figure.figsize'] = (12, 6)  # 修改为两个面板的宽度
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 设置scanpy参数
sc.settings.set_figure_params(dpi=300, frameon=True)

# 设置路径
base_dir = "."
data_path = os.path.join(base_dir, "data/raw/HNOCA.h5ad")
llm_results_path = os.path.join(base_dir, "results/benchmark/reference_comparison/2_evaluation/HNOCA_L3_results.csv")
output_dir = os.path.join(base_dir, "manuscript/figures")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取数据
print("Reading data...")
adata = sc.read_h5ad(data_path)
llm_results = pd.read_csv(llm_results_path)

# 创建颜色方案
color_scheme = {
    # NPCs (Blues to Purples)
    'Neural progenitor cells': '#4292C6',
    'Neural Progenitor Cells': '#4292C6',  # 添加大写版本
    'Neural Progenitors': '#4292C6',  # 添加别名
    'Cerebellar Neural Progenitor Cells': '#08519C',
    'Dorsal Midbrain Neural Progenitor Cells': '#2171B5',
    'Dorsal telencephalon neural progenitor cells': '#4292C6',
    'Dorsal Telencephalon Neural Progenitor Cells': '#4292C6',  # 添加大写版本
    'Dorsal Telencephalon Neural Stem Cells': '#4292C6',  # 添加别名
    'Dorsal Telencephalon Intermediate Progenitor Cell': '#6BAED6',
    'Hypothalamic Progenitors': '#7E57C2',
    'Neural Progenitor Cells (Medulla Specific)': '#9ECAE1',
    'Neural progenitor cells (Pons)': '#C6DBEF',
    'Interneuron progenitors (Thalamus)': '#9575CD',
    'Ventral Midbrain Progenitors': '#084594',
    'Ventromedial progenitors': '#084594',  # 添加别名
    'Ventral Telencephalon Neural Progenitor Cells': '#2171B5',
    'Neuroepithelium': '#4292C6',  # 添加原始参考名称
    'EC': '#4292C6',  # 添加原始参考名称

    # Neurons (Reds)
    'Cerebellar Granule Neurons': '#99000D',
    'Dorsal Midbrain Immature Neuron': '#CB181D',
    'Dorsal Midbrain Neurons': '#CB181D',  # 添加别名
    'Dorsal Telencephalon Projection Neurons': '#EF3B2C',
    'Excitatory Neurons (Dorsal Telencephalon Specific)': '#EF3B2C',  # 添加别名
    'Hypothalamic Excitatory Neurons': '#FB6A4A',
    'Medullary Neurons': '#FC9272',
    'Neurons (Pons)': '#FCBBA1',
    'Excitatory neurons of the Thalamus': '#C2185B',
    'Midbrain dopaminergic neurons': '#67000D',
    'Ventral Telencephalon Neuronal Progenitors': '#A50F15',

    # Glial lineage (Greens)
    'Astrocytes': '#00441B',
    'Radial Glia': '#9C27B0',
    'Radial glia': '#9C27B0',  # 添加小写版本
    'Oligodendrocyte Precursor Cells': '#238B45',
    'Oligodendrocyte precursor cells (OPCs)': '#238B45',  # 添加别名

    # Epithelial lineage (Purples)
    'Choroid plexus epithelial cells': '#54278F',
    'Choroid Plexus Epithelial Cell': '#54278F',  # 添加单数形式
    'Choroid Plexus Epithelial Cells': '#54278F',  # 添加大写版本
    'Hypothalamic Neuroepithelial Cells': '#673AB7',  # 使用更深的紫色
    'Early Epithelial Cells (Hypothalamus Specific)': '#673AB7',  # 添加别名
    'Epithelial cells': '#66C2A4',
    'Epithelial-like cells': '#4DB6AC',  # 青绿色，区别于普通上皮细胞

    # Immune lineage
    'Neural Macrophage/Microglia': '#8D6E63',
    'Microglia': '#8D6E63',  # 添加简化名称

    # Mesenchymal lineage (Browns)
    'Meningeal Fibroblasts': '#8C510A',
    'Fibroblasts': '#8C510A',  # 添加简化名称

    # Neural crest lineage (Teals)
    'Schwann cells': '#01665E',

    # Other cell types
    'GABAergic interneurons': '#D81B60'  # 紫红色，区别于其他神经元
}

# Function to calculate centroids
def calculate_centroids(coords, labels):
    unique_labels = np.unique(labels)
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            centroid = np.mean(coords[mask], axis=0)
            centroids[label] = centroid
    return centroids

# Set up the plotting style
sc.settings.set_figure_params(dpi=300, frameon=False)
plt.rcParams['figure.figsize'] = (18, 6)  # Width for three plots
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True

# Create standardized cell type names dictionary
standardized_names = {
    # Reference -> Standard
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
    'Neuroepithelium': 'Neural Progenitor Cells',
    'OPC': 'Oligodendrocyte Precursor Cells',
    'Pons NPC': 'Neural progenitor cells (Pons)',
    'Pons Neuron': 'Neurons (Pons)',
    'Thalamic NPC': 'Interneuron progenitors (Thalamus)',
    'Thalamic Neuron': 'Excitatory neurons of the Thalamus',
    'Ventral Midbrain NPC': 'Ventral Midbrain Progenitors',
    'Ventral Midbrain Neuron': 'Midbrain dopaminergic neurons',
    'Ventral Telencephalic NPC': 'Ventral Telencephalon Neural Progenitor Cells',
    'Ventral Telencephalic Neuron': 'Ventral Telencephalon Neuronal Progenitors',

    # 添加特殊处理的案例
    'Neural Progenitor Cells': 'Neural progenitor cells',
    'Neural Progenitors': 'Neural progenitor cells',
    'Choroid Plexus Epithelial Cell': 'Choroid plexus epithelial cells',
    'Choroid Plexus Epithelial Cells': 'Choroid plexus epithelial cells',
    'Dorsal Midbrain Neurons': 'Dorsal Midbrain Immature Neuron',
    'Dorsal Telencephalon Neural Progenitor Cells': 'Dorsal telencephalon neural progenitor cells',
    'Dorsal Telencephalon Neural Stem Cells': 'Dorsal telencephalon neural progenitor cells',
    'Early Epithelial Cells (Hypothalamus Specific)': 'Hypothalamic Neuroepithelial Cells',
    'Epithelial-like cells': 'Epithelial cells',
    'Excitatory Neurons (Dorsal Telencephalon Specific)': 'Dorsal Telencephalon Projection Neurons',
    'Fibroblasts': 'Meningeal Fibroblasts',
    'GABAergic interneurons': 'GABAergic interneurons',
    'Hypothalamic Neuroepithelial Cells': 'Hypothalamic Neuroepithelial Cells',
    'Microglia': 'Neural Macrophage/Microglia',
    'Oligodendrocyte precursor cells (OPCs)': 'Oligodendrocyte Precursor Cells',
    'Radial glia': 'Radial Glia',
    'Ventromedial progenitors': 'Ventral Midbrain Progenitors'
}

# Create UMAP plot
print("Plotting UMAPs...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Set figure style for each subplot
for ax in [ax1, ax2, ax3]:
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

# Add subplot labels and titles
ax1.text(-0.1, 1.1, 'a', transform=ax1.transAxes, fontsize=16, weight='bold')
ax2.text(-0.1, 1.1, 'b', transform=ax2.transAxes, fontsize=16, weight='bold')
ax3.text(-0.1, 1.1, 'c', transform=ax3.transAxes, fontsize=16, weight='bold')

# Add titles
ax1.text(0.5, 1.02, 'Reference', transform=ax1.transAxes, fontsize=14, ha='center')
ax2.text(0.5, 1.02, 'Our Framework', transform=ax2.transAxes, fontsize=14, ha='center')
ax3.text(0.5, 1.02, 'GPTCelltype', transform=ax3.transAxes, fontsize=14, ha='center')

# Add axis labels
ax1.set_xlabel('UMAP1', fontsize=12)
ax1.set_ylabel('UMAP2', fontsize=12)
ax2.set_xlabel('UMAP1', fontsize=12)
ax2.set_ylabel('UMAP2', fontsize=12)
ax3.set_xlabel('UMAP1', fontsize=12)
ax3.set_ylabel('UMAP2', fontsize=12)

# Get UMAP coordinates
umap_coords = adata.obsm['X_umap_scpoli']

# Function to standardize cell type names
def standardize_cell_type(cell_type):
    if cell_type is None:
        return cell_type

    # 应用standardized_names字典映射
    std_name = standardized_names.get(cell_type, cell_type)

    # 处理大小写不一致的情况
    for key, value in standardized_names.items():
        if cell_type.lower() == key.lower() and cell_type != key:
            return value

    return std_name

# Use original reference annotations
adata.obs['Reference'] = adata.obs['annot_level_3_rev2']

# Add LLMCellType predictions to adata
adata.obs['LLMCellType'] = 'Unknown'
adata.obs['GPTCelltype'] = 'Unknown'
for idx, row in llm_results.iterrows():
    ref_name = row['reference_name']
    final_consensus = row['final_consensus']
    gpt4o_prediction = row['initial_gpt-4o']
    mask = adata.obs['annot_level_3_rev2'] == ref_name
    adata.obs.loc[mask, 'LLMCellType'] = final_consensus
    adata.obs.loc[mask, 'GPTCelltype'] = gpt4o_prediction

# Analysis of cell type differences
print("\n=== Cell Type Mapping Analysis ===")

# Get unique values
ref_types = set(adata.obs['Reference'].unique())
llm_types = set(adata.obs['LLMCellType'].unique())
gpt_types = set(adata.obs['GPTCelltype'].unique())

# Remove 'Unknown' from predictions if present
if 'Unknown' in llm_types:
    llm_types.remove('Unknown')
if 'Unknown' in gpt_types:
    gpt_types.remove('Unknown')

# Create standardized sets
std_ref_types = {standardized_names.get(ct, ct) for ct in ref_types}
std_llm_types = {standardized_names.get(ct, ct) for ct in llm_types}
std_gpt_types = {standardized_names.get(ct, ct) for ct in gpt_types}

# Find unique and common cell types
only_in_ref = std_ref_types - std_llm_types
only_in_llm = std_llm_types - std_ref_types
common_types = std_ref_types & std_llm_types

# Find unique and common cell types for GPT-4o
only_in_ref_vs_gpt = std_ref_types - std_gpt_types
only_in_gpt = std_gpt_types - std_ref_types
common_types_gpt = std_ref_types & std_gpt_types

print("\n1. Cell types only in Reference Annotations:")
for ct in sorted(only_in_ref):
    original_types = [t for t in ref_types if standardized_names.get(t, t) == ct]
    count = sum(np.sum(adata.obs['Reference'] == t) for t in original_types)
    print(f"  - {ct} ({count} cells)")
    print(f"    Original names: {', '.join(original_types)}")

print("\n2. Cell types only in LLM Predictions:")
for ct in sorted(only_in_llm):
    original_types = [t for t in llm_types if standardized_names.get(t, t) == ct]
    count = sum(np.sum(adata.obs['LLMCellType'] == t) for t in original_types)
    print(f"  - {ct} ({count} cells)")
    print(f"    Original names: {', '.join(original_types)}")

print("\n3. Common cell types (with distribution):")
for ct in sorted(common_types):
    ref_original = [t for t in ref_types if standardized_names.get(t, t) == ct]
    llm_original = [t for t in llm_types if standardized_names.get(t, t) == ct]

    ref_count = sum(np.sum(adata.obs['Reference'] == t) for t in ref_original)
    llm_count = sum(np.sum(adata.obs['LLMCellType'] == t) for t in llm_original)

    print(f"\n  {ct}:")
    print(f"    Reference: {ref_count} cells")
    print(f"    LLM Prediction: {llm_count} cells")
    print(f"    Difference: {abs(ref_count - llm_count)} cells ({abs(ref_count - llm_count)/max(ref_count, llm_count)*100:.1f}%)")
    if ref_original != llm_original:
        print(f"    Reference names: {', '.join(ref_original)}")
        print(f"    LLM names: {', '.join(llm_original)}")

print(f"\n4. Summary Statistics:")
print(f"  Total unique cell types in Reference: {len(std_ref_types)}")
print(f"  Total unique cell types in LLM: {len(std_llm_types)}")
print(f"  Common cell types (LLM): {len(common_types)}")
print(f"  Matching rate (LLM): {len(common_types)/len(std_ref_types)*100:.1f}% of reference types")
print(f"  Total unique cell types in GPTCelltype: {len(std_gpt_types)}")
print(f"  Common cell types (GPTCelltype): {len(common_types_gpt)}")
print(f"  Matching rate (GPTCelltype): {len(common_types_gpt)/len(std_ref_types)*100:.1f}% of reference types")

# Print original distributions for reference
print("\n5. Original Reference Distribution:")
type_counts = adata.obs['Reference'].value_counts()
for cell_type, count in type_counts.items():
    print(f"{cell_type}: {count} cells")

# Plot Reference Annotations
texts = []
for ct in sorted(adata.obs['Reference'].unique()):
    mask = adata.obs['Reference'] == ct
    color = color_scheme.get(standardized_names.get(ct, ct), '#D3D3D3')
    if color == '#D3D3D3':
        print(f"Warning: No color defined for reference cell type: {ct}")
    ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                c=color, s=1, alpha=0.3, label=ct, rasterized=True)
    if np.sum(mask) > 100:  # Only show labels for clusters with more than 100 cells
        centroid = np.mean(umap_coords[mask], axis=0)
        # Use standardized name for the label
        label = standardized_names.get(ct, ct)
        texts.append(ax1.text(centroid[0], centroid[1], label,
                 fontsize=6, ha='center', va='center'))
adjust_text(texts, ax=ax1)

# Create a reverse mapping from standardized names to original names
reverse_mapping = {}
for orig, std in standardized_names.items():
    if std not in reverse_mapping:
        reverse_mapping[std] = orig

# Plot LLMCellType Annotations
texts = []
for ct in sorted(adata.obs['LLMCellType'].unique()):
    if ct != 'Unknown':
        mask = adata.obs['LLMCellType'] == ct
        # 先应用标准化再获取颜色
        std_name = standardized_names.get(ct, ct)
        color = color_scheme.get(std_name, color_scheme.get(ct, '#D3D3D3'))
        if color == '#D3D3D3':
            print(f"Warning: No color defined for predicted cell type: {ct} (standardized: {std_name})")
        ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                    c=color, s=1, alpha=0.3, label=ct, rasterized=True)
        if np.sum(mask) > 100:  # Only show labels for clusters with more than 100 cells
            centroid = np.mean(umap_coords[mask], axis=0)
            # 使用标准化的名称作为标签
            texts.append(ax2.text(centroid[0], centroid[1], std_name,
                     fontsize=6, ha='center', va='center'))
adjust_text(texts, ax=ax2)

# Plot GPTCelltype Annotations
texts = []
for ct in sorted(adata.obs['GPTCelltype'].unique()):
    if ct != 'Unknown':
        mask = adata.obs['GPTCelltype'] == ct
        # 先应用标准化再获取颜色
        std_name = standardized_names.get(ct, ct)
        color = color_scheme.get(std_name, color_scheme.get(ct, '#D3D3D3'))
        if color == '#D3D3D3':
            print(f"Warning: No color defined for GPTCelltype cell type: {ct} (standardized: {std_name})")
        ax3.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                    c=color, s=1, alpha=0.3, label=ct, rasterized=True)
        if np.sum(mask) > 100:  # Only show labels for clusters with more than 100 cells
            centroid = np.mean(umap_coords[mask], axis=0)
            # 使用标准化的名称作为标签
            texts.append(ax3.text(centroid[0], centroid[1], std_name,
                     fontsize=6, ha='center', va='center'))
adjust_text(texts, ax=ax3)

# Adjust layout
plt.tight_layout()

# Save figures
output_file = os.path.join(output_dir, 'supplementary_figure2.pdf')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
output_file = os.path.join(output_dir, 'supplementary_figure2.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"Figures saved to {output_dir}")
plt.close()
