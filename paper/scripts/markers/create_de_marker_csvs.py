import pandas as pd
import os

def create_de_marker_csv(df_tissue, output_file, top_n):
    """
    为指定组织创建差异表达标记基因的CSV文件
    df_tissue: 某个组织的差异表达数据
    output_file: 输出文件路径
    top_n: 每个细胞类型选择的top N个基因
    """
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 按细胞类型分组，为每个细胞类型选择top N个基因
    tissue_data = {}
    for cell_type in df_tissue['celltype'].unique():
        # 获取该细胞类型的所有基因
        cell_genes = df_tissue[df_tissue['celltype'] == cell_type]
        # 按p值和t统计量排序
        cell_genes = cell_genes.sort_values(['pvals', 'tstat'], ascending=[True, False])
        # 选择top N个基因，去除重复
        top_genes = cell_genes['gene'].drop_duplicates().head(top_n).tolist()
        tissue_data[cell_type] = top_genes

    # 转换为所需的CSV格式
    formatted_data = []
    for cell_type, genes in tissue_data.items():
        formatted_data.append({
            'cluster': cell_type,
            'gene': ','.join(genes)
        })

    # 直接写入文件
    with open(output_file, 'w') as f:
        f.write('cluster,gene\n')
        for item in formatted_data:
            f.write(f"{item['cluster']},{item['gene']}\n")
    print(f"Created {output_file}")

# 从Table S2读取数据
file_path = "data/reference/NIHMS1828607-supplement-Supplementary_Tables/science.abl4290_table_s2.xlsx"
df = pd.read_excel(file_path, skiprows=1)

# 处理每个组织
tissues = {
    'Breast': 'breast',
    'Esophagus mucosa': 'esophagus_mucosa',
    'Esophagus muscularis': 'esophagus_muscularis',
    'Heart': 'heart',
    'Lung': 'lung',
    'Prostate': 'prostate',
    'Skeletal muscle': 'skeletal_muscle',
    'Skin': 'skin'
}

# 为每个组织创建三个不同大小的标记基因集
for tissue_name, tissue_file in tissues.items():
    df_tissue = df[df['tissue'] == tissue_name]

    for top_n in [10, 20, 30]:
        output_file = f"data/reference/GTEx_DE_{tissue_file}_top{top_n}_markers.csv"
        create_de_marker_csv(df_tissue, output_file, top_n)
