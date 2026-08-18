import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

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
            r'Stress-activated'
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
sc.set_figure_params(figsize=(15, 12))

# 读取数据
print("Reading h5ad file...")
adata = sc.read("data/raw/HLCA_Core.h5ad")

# 读取注释结果
print("Reading annotation results...")
annotations = pd.read_csv("results/benchmark/popv_comparison/2_evaluation/HLCA_leiden_4_results.csv")

# 标准化注释
print("Standardizing annotations...")
annotations['standardized_consensus'] = annotations['final_consensus'].apply(standardize_cell_type)

# 打印标准化前后的对比
print("\nAnnotation standardization examples:")
for orig, stand in zip(annotations['final_consensus'].unique(), annotations['standardized_consensus'].unique()):
    if orig != stand:
        print(f"Original: {orig}")
        print(f"Standardized: {stand}\n")

# 创建cluster到标准化注释的映射
print("Creating mapping...")
cluster_to_annotation = dict(zip(annotations['reference_name'], annotations['standardized_consensus']))

# 将注释添加到adata对象中
print("Adding annotations to adata object...")
adata.obs['leiden_4'] = adata.obs['leiden_4'].astype(str)
adata.obs['LLM_annotations'] = adata.obs['leiden_4'].map(cluster_to_annotation)

# 设置matplotlib参数
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['figure.dpi'] = 300

# 生成颜色映射
print("Generating color palette...")
unique_annotations = sorted(adata.obs['LLM_annotations'].unique())
num_categories = len(unique_annotations)
colors = sns.husl_palette(num_categories, h=0.1)
color_dict = dict(zip(unique_annotations, colors))

# 绘制UMAP
print("Plotting UMAP...")
sc.pl.umap(
    adata,
    color='LLM_annotations',
    palette=color_dict,
    legend_loc='right margin',
    legend_fontsize=7,
    size=3,
    alpha=0.7,
    title='Cell Type Annotations (LLM)',
    frameon=False,
    show=False
)

# 保存图片
print("Saving plots...")
save_dir = 'results/benchmark/popv_comparison/3_visualization'
plt.savefig(os.path.join(save_dir, 'umap_standardized_annotations_v5.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(save_dir, 'umap_standardized_annotations_v5.png'), bbox_inches='tight', dpi=300)
plt.close()

# 保存标准化的注释结果
print("Saving standardized annotations...")
annotations.to_csv(os.path.join(save_dir, 'standardized_annotations_v5.csv'), index=False)

print("Done!")
