import pandas as pd
import os

def create_combined_marker_csv(df, output_file):
    """
    创建一个合并的标记基因CSV文件，每行包含一个细胞类型和它的所有标记基因

    参数：
    df: 包含标记基因的DataFrame
    output_file: 输出文件路径
    """
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 获取所有细胞类型
    cell_types = [col.replace('_marker', '') for col in df.columns if not pd.isna(col) and col.endswith('_marker')]

    # 创建输出数据
    rows = []
    for cell_type in cell_types:
        marker_col = f"{cell_type}_marker"
        genes = df[marker_col].dropna().tolist()
        # 将所有基因用逗号连接
        gene_str = ','.join(genes)
        # 每行的格式：cell_type,gene1,gene2,gene3,...
        rows.append(f"{cell_type},{gene_str}")

    # 写入CSV文件
    with open(output_file, 'w') as f:
        f.write('cluster,gene\n')  # 写入标题行
        for row in rows:
            f.write(f"{row}\n")
    print(f"Created {output_file}")

# 读取marker genes表
file_path = "data/processed/41591_2023_2327_MOESM3_ESM.xlsx"
df = pd.read_excel(file_path, sheet_name='6 - marker genes')

# 获取第一行作为列名
header = df.iloc[0]
df = df.iloc[1:]  # 删除第一行
df.columns = header  # 使用第一行作为列名

# 创建合并的CSV文件
output_file = "data/reference/HLCA_combined_markers.csv"
create_combined_marker_csv(df, output_file)
