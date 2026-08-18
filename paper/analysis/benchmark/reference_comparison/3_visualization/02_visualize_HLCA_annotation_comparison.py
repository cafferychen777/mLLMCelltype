import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import numpy as np

def get_cell_type_color_mapping(all_cell_types):
    """
    创建细胞类型到颜色的映射，确保相似的细胞类型使用相同的颜色
    """
    # 定义主要的细胞类型组
    cell_type_groups = {
        'AT': [
            'AT1', 'AT2', 'AT1 cells', 'AT2 cells', 'AT2 proliferating',
            'Transitional Club-AT2'
        ],
        'T_cells': [
            'T cell lineage', 'T cells', 'CD4 T cells', 'CD8 T cells',
            'T cells proliferating', 'Gamma Delta (γδ) T cells'
        ],
        'B_cells': [
            'B cell lineage', 'B cells', 'Plasma cells'
        ],
        'NK_cells': [
            'NK cells', 'Innate lymphoid cell NK'
        ],
        'Dendritic': [
            'Dendritic cells', 'DC1', 'DC2', 'Migratory DCs',
            'Plasmacytoid DCs'
        ],
        'Monocytes_Macrophages': [
            'Monocytes', 'Macrophages', 'Classical monocytes',
            'Non-classical monocytes', 'Alveolar macrophages',
            'Interstitial macrophages', 'Mast cells'
        ],
        'Endothelial': [
            'EC arterial', 'EC capillary', 'EC venous',
            'EC aerocyte capillary', 'EC general capillary',
            'EC venous pulmonary', 'EC venous systemic',
            'Endothelial cells', 'Lymphatic EC differentiating',
            'Lymphatic EC mature', 'Lymphatic EC proliferating'
        ],
        'Fibroblasts': [
            'Fibroblasts', 'Adventitial fibroblasts',
            'Alveolar fibroblasts', 'Peribronchial fibroblasts',
            'Subpleural fibroblasts', 'Myofibroblasts',
            'Myofibroblast cells'
        ],
        'Epithelial': [
            'Basal', 'Basal cells', 'Basal resting', 'Suprabasal',
            'Club', 'Club/Clara cells', 'Club/Goblet cells',
            'Goblet', 'Mucous cells', 'Mucous/goblet cells',
            'Secretory', 'Secretory epithelial cells',
            'Epithelial cells', 'Submucosal Secretory'
        ],
        'Ciliated': [
            'Multiciliated', 'Multiciliated lineage',
            'Ciliated cells', 'Deuterosomal'
        ],
        'Specialized': [
            'Ionocyte', 'Ionocytes cells',
            'Neuroendocrine', 'Neural cells',
            'Tuft', 'Hillock-like',
            'Pericytes', 'Pericytes cells'
        ],
        'Other': [
            'None', 'Rare', 'Low quality cells',
            'SMG duct', 'SMG mucous', 'SMG serous'
        ]
    }

    # 生成颜色映射
    colors = sns.husl_palette(len(cell_type_groups), h=0.1)
    color_dict = {}

    # 为每个组分配颜色
    for (group, cell_types), color in zip(cell_type_groups.items(), colors):
        for cell_type in cell_types:
            color_dict[cell_type] = color

    ungrouped = [ct for ct in all_cell_types if ct not in color_dict]
    if ungrouped:
        extra_colors = sns.husl_palette(len(ungrouped), h=0.5)  # 使用不同的色调范围
        for cell_type, color in zip(ungrouped, extra_colors):
            color_dict[cell_type] = color

    return color_dict

def fill_none_with_upper_level(adata):
    """
    Fill None values in each level with non-None values from upper levels.
    For example, if level_4 is None, try level_3, then level_2, then level_1.
    Also handles cases where 'None' is a string value.
    """
    level_columns = ['ann_level_1', 'ann_level_2', 'ann_level_3', 'ann_level_4', 'ann_level_5']

    for i in range(len(level_columns)-1, -1, -1):
        current_col = level_columns[i]
        if current_col not in adata.obs.columns:
            continue

        # Convert categorical to string type for manipulation
        current_series = adata.obs[current_col].astype(str)
        none_mask = current_series.isin(['None', 'nan'])

        if none_mask.any():
            filled_values = current_series.copy()

            # Try to fill from upper levels
            for j in range(i-1, -1, -1):
                upper_col = level_columns[j]
                if upper_col not in adata.obs.columns:
                    continue

                upper_series = adata.obs[upper_col].astype(str)
                still_none = filled_values.isin(['None', 'nan'])

                if still_none.any():
                    # Only fill values that are still None
                    upper_values = upper_series[still_none]
                    valid_mask = ~upper_values.isin(['None', 'nan'])
                    if valid_mask.any():
                        filled_values.loc[still_none][valid_mask] = upper_values[valid_mask]

            # Convert back to categorical with original categories
            original_categories = adata.obs[current_col].cat.categories
            filled_values = pd.Categorical(filled_values, categories=original_categories)
            adata.obs[current_col] = filled_values

    return adata

def standardize_cell_type(cell_type):
    if pd.isna(cell_type):
        return cell_type

    # 标准化映射字典
    standardization_map = {
        'AT2 cells': [
            r'Alveolar Type 2', r'Type 2 Alveolar', r'Alveolar Type II',
            r'Type II pneumocytes', r'AT2', r'Type 2 cells', r'Type II cells',
            r'Alveolar Epithelial Type II', r'Alveolar Type 2 Epithelial'
        ],
        'AT1 cells': [
            r'Alveolar Type 1', r'Type 1 Alveolar', r'Alveolar Type I',
            r'Type I cells', r'AT1', r'Type 1 cells',
            r'Alveolar Epithelial Type I', r'Alveolar Type 1 Epithelial'
        ],
        'NK cells': [
            r'NK cells?',
            r'NK and T',
            r'NK cells? and cytotoxic lymphocytes?',
            r'Natural [Kk]iller'
        ],
        'T cells': [
            r'CD4\+ T', r'CD8\+ T', r'Cytotoxic CD8\+ T', r'Activated T',
            r'T/NK', r'T and NK', r'Gamma Delta \(yõ\) T'
        ],
        'Neural cells': [
            r'Neural cells?',
            r'Neuroendocrine cells?'
        ],
        'Proliferating cells': [
            r'Proliferating cells?',
            r'Proliferating [Ee]pithelial',
            r'Proliferating [Mm]yeloid',
            r'Cluster \d+:\s*Proliferating'
        ],
        'Secretory epithelial cells': [
            r'Secretory cells?',
            r'Secretory [Ee]pithelial',
            r'Secretory/WFDC2\+',
            r'Salivary gland acinar'
        ],
        'Muscle cells': [
            r'Skeletal [Mm]uscle',
            r'Smooth [Mm]uscle',
            r'Myofibroblast[- ]like'
        ],
        'Plasma cells': [
            r'Plasma [Cc]ells?',
            r'Plasmacytoid Dendritic'
        ],
        'Ribosomal cells': [
            r'Ribosomal protein[- ]rich',
            r'Ribosomal proteins?.*?(?:stromal|proliferative)',
            r'Ribosomal/Translation(?:[- ]active)?',
            r'Ribosomal/Housekeeping'
        ],
        'Stress response cells': [
            r'Stress [Rr]esponse',
            r'Stress[- ]responsive',
            r'Stressed/Heat Shock Response',
            r'Stress activated',
            r'Stressed',
            r'Stress-activated',
            r'SM activated stress response'
        ],
        'Stromal cells': [
            r'Stromal or myofibroblast[- ]like',
            r'Stromal cells?'
        ],
        'Keratinocytes': [
            r'Suprabasal/Differentiated [Kk]eratinocyte',
            r'Keratinocyte cells?'
        ],
        'Neutrophils': [
            r'Neutrophils? cells?'
        ],
        'Pericytes': [
            r'Pericytes? cells?'
        ],
        'B cells': [
            r'B cells or plasma', r'B cell', r'B lymphocytes?'
        ],
        'Monocytes': [
            r'\bMonocytes?\b(?:\s+cells?)?',
            r'Inflammatory monocytes?',
            r'Monocytes?/Dendritic',
            r'Monocytes?/Macrophages?'
        ],
        'Macrophages': [
            r'\bMacrophages?\b(?:\s+cells?)?',
            r'Inflammatory Macrophages?(?:\s+cells?)?',
            r'Pro-inflammatory Macrophages?',
            r'Alternative macrophages?',
            r'Interstitial Macrophages?',
            r'Activated Macrophages?',
            r'Alveolar Macrophages?'
        ],
        'Mesenchymal cells': [
            r'Mesenchymal cells?',
            r'Mesenchymal progenitor',
            r'Mesothelial cells?',
            r'Myofibroblast cells?'
        ],
        'Mitochondrial-rich cells': [
            r'Mitochondrial[- ]Rich Endothelial',
            r'Mitochondrial[- ]rich cells?',
            r'Mitochondrial/Metabolic',
            r'Mitochondrial and Myeloid',
            r'Mitochondrial genes/low quality'
        ],
        'Low quality cells': [
            r'Low quality cells?',
            r'Mixed/Low Quality Cluster',
            r'Mixed epithelial cell types?',
            r'Mitochondrial genes/low quality'
        ],
        'Heat shock protein-expressing cells': [
            r'Heat shock protein', r'Heat-shock protein', r'HSP'
        ],
        'Immediate early response cells': [
            r'Immediate Early Response', r'IER'
        ],
        'Antigen presenting cells': [
            r'Antigen presenting', r'Immune/antigen presenting',
            r'Antigen-presenting'
        ],
        'Basal cells': [
            r'Basal Cells?', r'Basal Epithelial Cells?',
            r'Basal cell', r'Basal Epithelial cell',
            r'Serpin-expressing basal'
        ],
        'Endothelial cells': [
            r'Capillary Endothelial', r'Inflammatory Endothelial',
            r'Stressed Endothelial', r'Mixed Endothelial',
            r'Endothelial Cells?', r'Activated endothelial',
            r'Lymphatic Endothelial'
        ],
        'Fibroblasts': [
            r'\bFibroblasts?\b(?:\s+cells?)?',
            r'Fibroblast and Mesenchymal',
            r'Activated Fibroblast',
            r'Stressed Fibroblast',
            r'Activated/Stressed Fibroblast'
        ],
        'Ciliated cells': [
            r'Ciliated Cells?', r'Multi-?ciliated cells?',
            r'Multiciliated cells?'
        ],
        'Epithelial cells': [
            r'Epithelial Secretory', r'Epithelial progenitor',
            r'Epithelial cells?'
        ],
        'Erythroid cells': [
            r'Erythrocytes?', r'Erythroid cells?'
        ],
        'Mast cells': [
            r'Mast cells?'
        ],
        'Metallothionein-expressing cells': [
            r'Metallothionein-expressing cells?'
        ],
        'Mucous cells': [
            r'Mucous cells?',
            r'Mucus[- ]secreting cells?'
        ],
        'Myeloid cells': [
            r'Myeloid cells?',
            r'Mitochondrial and Myeloid'
        ],
        'Interferon-response cells': [
            r'Interferon-response cells?',
            r'IFN[- ]response cells?'
        ],
        'Leukocytes': [
            r'Leukocytes? cells?',
            r'Ionocytes? cells?'
        ],
        'Male-specific immune cells': [
            r'Male-specific immune cells?'
        ]
    }

    # 移除开头的数字和特殊字符
    cell_type = re.sub(r'^\d+[:.]\s*', '', cell_type)
    cell_type = re.sub(r'^[•]\s*', '', cell_type)
    cell_type = cell_type.strip()

    # 标准化名称
    for standard_name, patterns in standardization_map.items():
        for pattern in patterns:
            if re.search(pattern, cell_type, re.IGNORECASE):
                return standard_name

    # 处理未匹配的情况
    # 确保以"cells"结尾
    if not cell_type.lower().endswith('cells') and not cell_type.lower().endswith('cell'):
        cell_type += ' cells'
    # 统一使用复数形式
    if cell_type.lower().endswith('cell'):
        cell_type = cell_type[:-4] + 'cells'

    # 确保首字母大写
    cell_type = cell_type[0].upper() + cell_type[1:]

    return cell_type

# 设置绘图参数
sc.set_figure_params(figsize=(20, 15))

# 读取数据
print("Reading h5ad file...")
adata = sc.read("data/raw/HLCA_Core.h5ad")

# 填充None值
print("Filling None values with upper level annotations...")
adata = fill_none_with_upper_level(adata)

# 读取LLM注释结果
print("Reading LLM annotation results...")
llm_annotations = pd.read_csv("results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_4_results.csv")

# 标准化LLM注释
print("Standardizing LLM annotations...")
llm_annotations['standardized_consensus'] = llm_annotations['final_consensus'].apply(standardize_cell_type)

# 创建cluster到标准化注释的映射
cluster_to_annotation = dict(zip(llm_annotations['reference_name'], llm_annotations['standardized_consensus']))

# 将LLM注释添加到adata对象中
print("Adding LLM annotations to adata object...")
adata.obs['leiden_4'] = adata.obs['leiden_4'].astype(str)
adata.obs['LLM_annotations'] = adata.obs['leiden_4'].map(cluster_to_annotation)

# 收集所有的细胞类型
all_cell_types = set()
for obs_key in ['ann_level_3', 'ann_level_4', 'LLM_annotations']:
    if obs_key in adata.obs.columns:
        all_cell_types.update(adata.obs[obs_key].unique())

# 获取颜色映射
color_dict = get_cell_type_color_mapping(list(all_cell_types))

# 设置matplotlib参数
plt.rcParams['figure.figsize'] = (20, 30)
plt.rcParams['figure.dpi'] = 300

# 创建一个3x1的子图布局
fig, axes = plt.subplots(3, 1, figsize=(20, 30))

# 打印每个注释级别的统计信息
print("\nAnnotation statistics before plotting:")
for obs_key in ['ann_level_3', 'ann_level_4', 'LLM_annotations']:
    if obs_key in adata.obs.columns:
        n_unique = len(adata.obs[obs_key].unique())
        n_none = adata.obs[obs_key].isna().sum()
        print(f"{obs_key}:")
        print(f"  - Unique annotations: {n_unique}")
        print(f"  - None values: {n_none}")
        print(f"  - Unique values: {sorted(adata.obs[obs_key].unique())}\n")

# 生成颜色映射并绘制UMAP
print("Plotting UMAPs...")
for obs_key, ax in zip(['ann_level_3', 'ann_level_4', 'LLM_annotations'], axes):
    if obs_key in adata.obs.columns:
        # 再次应用填充逻辑
        current_series = adata.obs[obs_key].copy()
        if pd.api.types.is_categorical_dtype(current_series):
            current_series = current_series.astype(str)
        none_mask = current_series.isin(['None', 'nan', 'NaN'])

        if none_mask.any():
            # 从上层注释中获取值
            level_columns = ['ann_level_1', 'ann_level_2', 'ann_level_3', 'ann_level_4']
            current_level_idx = level_columns.index(obs_key) if obs_key in level_columns else -1

            if current_level_idx > 0:  # 如果有上层注释可用
                for upper_level in level_columns[:current_level_idx]:
                    if upper_level in adata.obs.columns:
                        upper_values = adata.obs[upper_level].astype(str)
                        still_none = none_mask & ~upper_values.isin(['None', 'nan', 'NaN'])
                        current_series.loc[still_none] = upper_values.loc[still_none]
                        none_mask = current_series.isin(['None', 'nan', 'NaN'])
                        if not none_mask.any():
                            break

        # 更新adata对象中的注释
        adata.obs[obs_key] = current_series

        # 确保所有的细胞类型都有颜色映射
        unique_types = adata.obs[obs_key].unique()
        for cell_type in unique_types:
            if cell_type not in color_dict:
                # 为未映射的类型生成新的颜色
                new_color = sns.husl_palette(1, h=0.1)[0]
                color_dict[cell_type] = new_color

        # 绘制UMAP
        print(f"Plotting UMAP for {obs_key}...")
        sc.pl.umap(
            adata,
            color=obs_key,
            palette=color_dict,
            legend_loc='right margin',
            legend_fontsize=6,
            size=3,
            alpha=0.7,
            title=f'Cell Type Annotations ({obs_key})',
            frameon=False,
            show=False,
            ax=ax
        )

# 调整布局
plt.tight_layout()

# 保存图片
print("Saving plots...")
save_dir = 'results/benchmark/popv_comparison/3_visualization'
plt.savefig(os.path.join(save_dir, 'umap_annotation_comparison_v4.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(save_dir, 'umap_annotation_comparison_v4.png'), bbox_inches='tight', dpi=300)
plt.close()

# 保存注释比较结果
print("Saving annotation comparison...")
comparison_df = pd.DataFrame({
    'leiden_4': adata.obs['leiden_4'],
    'Level_3': adata.obs['ann_level_3'],
    'Level_4': adata.obs['ann_level_4'],
    'LLM_annotations': adata.obs['LLM_annotations']
})
comparison_df.to_csv(os.path.join(save_dir, 'annotation_comparison_v4.csv'), index=True)

print("Done!")
